/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Diag.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: chris
 */

#include "Diag.h"
#include <chrono>
#include <ctype.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define DCGM_INIT_UUID
#include "DcgmDiagCommon.h"
#include "DcgmStringHelpers.h"
#include "DcgmUtilities.h"
#include "NvcmTCLAP.h"
#include "NvvsJsonStrings.h"
#include "PluginStrings.h"
#include "dcgm_agent.h"
#include "dcgm_errors.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include "dcgm_test_apis.h"
#include <DcgmBuildInfo.hpp>

/* Process Info */
char DIAG_HEADER[] = "+---------------------------+------------------------------------------------+\n"
                     "| Diagnostic                | Result                                         |\n"
                     "+===========================+================================================+\n";

char DIAG_DATA[] = "| <DATA_NAME              > | <DATA_INFO                                   > |\n";

const char DIAG_INFO[] = "|-----  Metadata  ----------+------------------------------------------------|\n";

char DIAG_FOOTER[] = "+---------------------------+------------------------------------------------+\n";

char DIAG_DEPLOYMENT[] = "|-----  Deployment  --------+------------------------------------------------|\n";

char DIAG_HARDWARE[] = "+-----  Hardware  ----------+------------------------------------------------+\n";

char DIAG_INTEGRATION[] = "+-----  Integration  -------+------------------------------------------------+\n";

char DIAG_STRESS[] = "+-----  Stress  ------------+------------------------------------------------+\n";

// Header Names
const std::string DISPLAY_DEPLOYMENT("Deployment");
const std::string DISPLAY_HARDWARE("Hardware");
const std::string DISPLAY_INTEGRATION("Integration");
const std::string DISPLAY_STRESS("Stress");

// Test Names
const std::string DISPLAY_DENYLIST("Denylist");
const std::string DISPLAY_NVML_LIB("NVML Library");
const std::string DISPLAY_CUDA_MAIN_LIB("CUDA Main Library");
const std::string DISPLAY_CUDA_TOOLKIT("CUDA Toolkit Library");
const std::string DISPLAY_PERMISSIONS("Permissions and OS Blocks");
const std::string DISPLAY_PERSISTENCE("Persistence Mode");
const std::string DISPLAY_ENVIRONMENT("Environment Variables");
const std::string DISPLAY_PAGE_RETIREMENT("Page Retirement/Row Remap");
const std::string DISPLAY_GRAPHICS("Graphics Processes");
const std::string DISPLAY_INFOROM("Inforom");

// Must follow the same order as dcgmSoftwareTest_enum in dcgm_structs.h
const std::string levelOneTests[] = {
    DISPLAY_DENYLIST,    DISPLAY_NVML_LIB,    DISPLAY_CUDA_MAIN_LIB,   DISPLAY_CUDA_TOOLKIT, DISPLAY_PERMISSIONS,
    DISPLAY_PERSISTENCE, DISPLAY_ENVIRONMENT, DISPLAY_PAGE_RETIREMENT, DISPLAY_GRAPHICS,     DISPLAY_INFOROM,
};


const std::string DISPLAY_MEMORY("GPU Memory");
const std::string DISPLAY_CTXCREATE("Context Create");
const std::string DISPLAY_SM_STRESS("SM Stress");
const std::string DISPLAY_TP("Targeted Power");
const std::string DISPLAY_TS("Targeted Stress");
const std::string DISPLAY_DIAGNOSTIC("Diagnostic");
const std::string DISPLAY_PCIE("PCIe");
const std::string DISPLAY_MEMBW("Memory Bandwidth");
const std::string DISPLAY_MEMTEST("Memtest");
const std::string DISPLAY_PULSE_TEST("Pulse Test");
const std::string DISPLAY_EUD_TEST("EUD Test");

#define DATA_NAME_TAG "<DATA_NAME"
#define DATA_INFO_TAG "<DATA_INFO"
// The amount of space for information
#define DATA_INFO_TAG_LEN 45

std::ifstream::pos_type filesize(const std::string &filename)
{
    std::ifstream in(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

/*******************************************************************************/
/* Variables for terminating a running diag on a SIGINT and other signals */
// To avoid conflicting with STL reserved namespace, variables are prefixed with diag_
static std::atomic<bool> g_signalExit = false;
static bool diag_stopDiagOnSignal     = false; // Whether we should attempt to stop a running diag on recieving a signal
static std::string diag_hostname      = "";    // Hostname for the remote host engine
static std::string diag_pathToExecutable = ""; // Path to the dcgmi executable
static bool diag_installed_sig_handlers  = false; // Whether sig handlers have been installed
static void (*diag_oldHupHandler)(int)   = NULL;  // reference to old sig hup handler if any
static void (*diag_oldIntHandler)(int)   = NULL;  // reference to old sig int handler if any
static void (*diag_oldQuitHandler)(int)  = NULL;  // reference to old sig quit handler if any
static void (*diag_oldTermHandler)(int)  = NULL;  // reference to old sig term handler if any

static struct sigaction diag_actHup, diag_actInt, diag_actQuit, diag_actTerm,
    diag_actOld; // sigaction structs for the handlers

/* Defines for sig handling */
// Concatenate given args into one token
#define CONCAT2(a, b)   a##b
#define CONCAT(a, b, c) a##b##c
// Run old signal handler
#define RUN_OLD_HANDLER(name, signum, handler)           \
    if (CONCAT(diag_old, name, Handler) == SIG_DFL)      \
    {                                                    \
        signal(signum, SIG_DFL);                         \
        raise(signum);                                   \
        signal(signum, handler);                         \
    }                                                    \
    else if (CONCAT(diag_old, name, Handler) != SIG_IGN) \
    CONCAT(diag_old, name, Handler)(signum)

#define SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIG, name, handler)          \
    /* Create new handler */                                              \
    memset(&CONCAT2(diag_act, name), 0, sizeof(CONCAT2(diag_act, name))); \
    CONCAT2(diag_act, name).sa_handler = handler;                         \
    sigemptyset(&CONCAT2(diag_act, name).sa_mask);                        \
    sigaddset(&CONCAT2(diag_act, name).sa_mask, SIG);                     \
    sigaction(SIG, &CONCAT2(diag_act, name), &diag_actOld);               \
    /* Save old handler */                                                \
    CONCAT(diag_old, name, Handler) = diag_actOld.sa_handler

/* Sig handler methods */
void handle_signal_during_diag(int signum)
{
    if (diag_stopDiagOnSignal)
    {
        g_signalExit = true;
    }
}

void InstallSigHandlers()
{
    // Ensure this method is called only once
    if (diag_installed_sig_handlers)
    {
        return;
    }
    diag_installed_sig_handlers = true;
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGHUP, Hup, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGINT, Int, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGQUIT, Quit, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGTERM, Term, handle_signal_during_diag);
}

/*****************************************************************************
 *****************************************************************************
 * RemoteDiagExecutor
 *****************************************************************************
 *****************************************************************************/
RemoteDiagExecutor::RemoteDiagExecutor(dcgmHandle_t handle, dcgmRunDiag_t &drd)
    : m_handle(handle)
    , m_result(DCGM_ST_OK)
{
    memcpy(&m_drd, &drd, sizeof(m_drd));
    memset(&m_diagResult, 0, sizeof(m_diagResult));
    m_diagResult.version = dcgmDiagResponse_version;
}

void RemoteDiagExecutor::run()
{
    m_result = dcgmActionValidate_v2(m_handle, &m_drd, &m_diagResult);
}

dcgmReturn_t RemoteDiagExecutor::GetResult() const
{
    return m_result;
}

dcgmDiagResponse_t RemoteDiagExecutor::GetResponse() const
{
    return m_diagResult;
}


/*****************************************************************************
 *****************************************************************************
 * Diag
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
Diag::Diag(unsigned int iterations, const std::string &hostname)
    : m_jsonOutput(false)
    , m_iterations(iterations)
    , m_hostname(hostname)
{
    memset(&this->m_drd, 0, sizeof(this->m_drd));
}

Diag::~Diag()
{}

/*******************************************************************************/
void Diag::setDcgmRunDiag(dcgmRunDiag_t *drd)
{
    memcpy(&this->m_drd, drd, sizeof(this->m_drd));
}

/*******************************************************************************/
void Diag::setJsonOutput(bool jsonOutput)
{
    this->m_jsonOutput = jsonOutput;
}

/*******************************************************************************/
dcgmReturn_t Diag::GetFailureResult(dcgmDiagResponse_t &diagResult)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    for (unsigned int i = 0; i < diagResult.levelOneTestCount; i++)
    {
        if (diagResult.levelOneResults[i].status == DCGM_DIAG_RESULT_FAIL)
        {
            for (unsigned int j = 0; j < DCGM_MAX_ERRORS; j++)
            {
                if (dcgmErrorGetPriorityByCode(diagResult.levelOneResults[i].error[j].code) == DCGM_ERROR_ISOLATE)
                {
                    return DCGM_ST_NVVS_ISOLATE_ERROR;
                }
                else
                {
                    ret = DCGM_ST_NVVS_ERROR;
                }
            }
        }
    }

    /* need to search through all devices because results are written to gpu indexes */
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; j++)
        {
            if (diagResult.perGpuResponses[i].results[j].status == DCGM_DIAG_RESULT_FAIL)
            {
                for (unsigned int k = 0; k < DCGM_MAX_ERRORS; k++)
                {
                    if (dcgmErrorGetPriorityByCode(diagResult.perGpuResponses[i].results[j].error[k].code)
                        == DCGM_ERROR_ISOLATE)
                    {
                        return DCGM_ST_NVVS_ISOLATE_ERROR;
                    }
                    else
                    {
                        ret = DCGM_ST_NVVS_ERROR;
                    }
                }
            }
        }
    }

    return ret;
}

void Diag::PopulateGpuList(const dcgmDiagResponse_t &diagResult, std::vector<unsigned int> &gpuVec)
{
    // No specified list; find the gpuIds that have been set with tests that have run
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES && gpuVec.size() < diagResult.gpuCount; i++)
    {
        if (diagResult.perGpuResponses[i].gpuId != DCGM_MAX_NUM_DEVICES)
        {
            bool someTestRan = false;
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; j++)
            {
                if (diagResult.perGpuResponses[i].results[j].status != DCGM_DIAG_RESULT_NOT_RUN)
                {
                    someTestRan = true;
                    break;
                }
            }

            if (someTestRan == true)
                gpuVec.push_back(i);
        }
    }
}

void Diag::InitializeDiagResponse(dcgmDiagResponse_t &diagResult)
{
    memset(&diagResult, 0, sizeof(diagResult));
    diagResult.version = dcgmDiagResponse_version;

    // Initialize the gpu id to one we know won't exist so we can figure out which
    // GPUs ran if there was no specified list
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
        diagResult.perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
}

void Diag::HelperDisplayFailureMessage(const std::string &errMsg, dcgmReturn_t result)
{
    if (m_jsonOutput)
    {
        Json::Value output;
        output[NVVS_NAME][NVVS_VERSION_STR]   = std::string(DcgmNs::DcgmBuildInfo().GetVersion());
        output[NVVS_NAME][NVVS_RUNTIME_ERROR] = errMsg;
        std::cout << output.toStyledString() << std::endl;
    }
    else
    {
        std::cout << errMsg << std::endl;
    }

    if (result != DCGM_ST_OK)
    {
        log_error("Error in diagnostic for group with ID: {}. Return: {} '{}'",
                  (unsigned int)(uintptr_t)m_drd.groupId,
                  result,
                  errMsg);
    }
}

/*******************************************************************************/
dcgmReturn_t Diag::RunDiagOnce(dcgmHandle_t handle)
{
    std::unique_ptr<dcgmDiagResponse_t> diagResultUPtr = std::make_unique<dcgmDiagResponse_t>();

    dcgmDiagResponse_t &diagResult = *(diagResultUPtr.get());
    dcgmReturn_t result = DCGM_ST_OK;
    std::vector<unsigned int> gpuVec;
    std::vector<std::string> gpuStrList;

    InitializeDiagResponse(diagResult);

    // Setup signal handlers
    InstallSigHandlers();

    result = ExecuteDiagOnServer(handle, diagResult);

    if (result == DCGM_ST_GROUP_INCOMPATIBLE)
    {
        HelperDisplayFailureMessage("Error: Diagnostic can only be performed on a homogeneous group of GPUs.", result);
        return result;
    }
    else if (result == DCGM_ST_NOT_SUPPORTED)
    {
        HelperDisplayFailureMessage(
            "Error: Diagnostic could not be run because the Tesla recommended driver is not being used.", result);
        return result;
    }
    else if (result == DCGM_ST_PAUSED)
    {
        HelperDisplayFailureMessage("Error: Diagnostic could not be run while DCGM is paused.", result);
        return result;
    }
    else if (result != DCGM_ST_OK)
    {
        std::stringstream errMsg;
        if (diagResult.systemError.msg[0] != '\0')
        {
            errMsg << diagResult.systemError.msg;
        }
        else
        {
            errMsg << "Error: Unable to complete diagnostic for group " << (unsigned int)(uintptr_t)m_drd.groupId
                   << ". Return: (" << result << ") " << errorString(result) << ".";
        }

        if (result == DCGM_ST_TIMEOUT)
        {
            // If there was a timeout, we attempt to stop the launched diagnostic before returning.
            dcgmReturn_t ret = dcgmStopDiagnostic(handle);
            if (ret != DCGM_ST_OK)
            {
                errMsg << "\nError: Could not stop the launched diagnostic.";
                log_error("There was an error stopping the launched diagnostic. Return: {}", ret);
            }
        }

        HelperDisplayFailureMessage(errMsg.str(), result);
        return result;
    }
    else if (m_jsonOutput == false && diagResult.systemError.msg[0] != '\0')
    {
        std::stringstream errMsg;
        errMsg << "Error: " << diagResult.systemError.msg << std::endl;
        HelperDisplayFailureMessage(errMsg.str(), result);
        return DCGM_ST_NVVS_ERROR;
    }

    if (strlen(m_drd.gpuList) > 0)
    {
        dcgmTokenizeString(m_drd.gpuList, ",", gpuStrList);
        for (size_t i = 0; i < gpuStrList.size(); i++)
        {
            gpuVec.push_back(strtol(gpuStrList[i].c_str(), NULL, 10));
        }
    }
    else
    {
        PopulateGpuList(diagResult, gpuVec);
    }

    if (m_jsonOutput)
    {
        result = HelperDisplayAsJson(diagResult, gpuVec);
    }
    else
    {
        std::cout << "Successfully ran diagnostic for group." << std::endl;

        std::cout << DIAG_HEADER;

        HelperDisplayVersionAndDevIds(diagResult);

        HelperDisplayDeployment(diagResult);

        if (gpuVec.size() > 0)
        {
            HelperDisplayIntegration(diagResult.perGpuResponses, gpuVec);
            HelperDisplayHardware(diagResult.perGpuResponses, gpuVec);
            HelperDisplayPerformance(diagResult.perGpuResponses, gpuVec);
        }

        std::cout << DIAG_FOOTER;
    }

    if (result == DCGM_ST_OK)
    {
        result = GetFailureResult(diagResult);
    }

    return result;
}

/*******************************************************************************/
dcgmReturn_t Diag::ExecuteDiagOnServer(dcgmHandle_t handle, dcgmDiagResponse_t &diagResult)
{
    std::unique_ptr<RemoteDiagExecutor> rde = std::make_unique<RemoteDiagExecutor>(handle, m_drd);
    dcgmReturn_t result   = DCGM_ST_OK;
    diag_stopDiagOnSignal = true;

    // Start the diagnostic
    rde->Start();

    while (1)
    {
        if (g_signalExit)
        {
            AbortDiag ad(m_hostname);
            ad.Execute();
            rde->Stop();
            result = DCGM_ST_NVVS_KILLED;
            break;
        }
        else if (rde->HasExited())
        {
            result     = rde->GetResult();
            diagResult = rde->GetResponse();
            break;
        }

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
    }

    // Reset global flag so that the sig handler does not attempt to stop a diag when no diag is running.
    diag_stopDiagOnSignal = false;

    return result;
}

/*******************************************************************************/
dcgmReturn_t Diag::RunStartDiag(dcgmHandle_t handle)

{
    if (m_iterations <= 1)
    {
        return RunDiagOnce(handle);
    }

    Json::Value output;
    dcgmReturn_t overallRet = DCGM_ST_OK;
    m_drd.totalIterations   = m_iterations;

    for (unsigned int i = 0; i < m_iterations; i++)
    {
        if (m_jsonOutput == false)
        {
            std::cout << "\nRunning iteration " << i + 1 << " of " << m_iterations << "...\n";
        }

        m_drd.currentIteration = i;

        dcgmReturn_t ret                             = RunDiagOnce(handle);
        output[NVVS_ITERATIONS][static_cast<int>(i)] = m_jsonTmpValue;

        if (ret != DCGM_ST_OK)
        {
            overallRet = ret;
            // Break out of the loop due to a failure
            if (m_jsonOutput == true)
            {
                output[NVVS_RESULT]  = "Fail";
                output[NVVS_WARNING] = errorString(ret);
            }
            break;
        }
    }

    if (m_jsonOutput == true)
    {
        if (overallRet == DCGM_ST_OK)
        {
            output[NVVS_RESULT] = "Pass";
        }

        std::cout << output.toStyledString() << std::endl;
    }
    else if (overallRet == DCGM_ST_OK)
    {
        std::cout << "Passed all " << m_iterations << " runs of the diagnostic" << std::endl;
    }
    else
    {
        std::cout << "Aborting the iterative runs of the diagnostic due to failure: " << errorString(overallRet)
                  << std::endl;
    }

    return overallRet;
}

dcgmReturn_t Diag::RunViewDiag()
{
    return DCGM_ST_OK;
}

void Diag::HelperDisplayDeploymentResult(CommandOutputController &cmdView,
                                         const std::string &nameTag,
                                         dcgmDiagTestResult_v3 &result)
{
    if (result.status != DCGM_DIAG_RESULT_NOT_RUN)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, nameTag);
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperDisplayDiagResult(result.status));
        cmdView.display();
        for (unsigned int i = 0; i < DCGM_MAX_ERRORS; i++)
        {
            if (result.error[i].msg[0] != '\0')
                DisplayVerboseInfo(cmdView, "Error", result.error[i].msg);
        }

        if (result.info[0] != '\0')
            DisplayVerboseInfo(cmdView, "Info", result.info);
    }
}

void Diag::HelperDisplayVersionAndDevIds(dcgmDiagResponse_t &diagResult)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_INFO;

    cmdView.setDisplayStencil(DIAG_DATA);
    cmdView.addDisplayParameter(DATA_NAME_TAG, "DCGM Version");
    cmdView.addDisplayParameter(DATA_INFO_TAG, diagResult.dcgmVersion);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Driver Version Detected");
    cmdView.addDisplayParameter(DATA_INFO_TAG, diagResult.driverVersion);
    cmdView.display();

    std::stringstream ss;

    for (unsigned int i = 0; i < diagResult.gpuCount; i++)
    {
        if (i == 0)
        {
            ss << diagResult.devIds[i];
        }
        else
        {
            ss << "," << diagResult.devIds[i];
        }
    }

    cmdView.addDisplayParameter(DATA_NAME_TAG, "GPU Device IDs Detected");
    cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
    cmdView.display();
}

void Diag::HelperDisplayDeployment(dcgmDiagResponse_t &diagResult)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_DEPLOYMENT;

    cmdView.setDisplayStencil(DIAG_DATA);

    for (unsigned int i = 0; i < diagResult.levelOneTestCount; i++)
    {
        HelperDisplayDeploymentResult(cmdView, levelOneTests[i], diagResult.levelOneResults[i]);
    }
}

void Diag::HelperDisplayHardware(dcgmDiagResponsePerGpu_v5 *diagResults, const std::vector<unsigned int> &gpuIndices)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_HARDWARE;

    if (!strcasecmp(m_drd.testNames[0], CTXCREATE_PLUGIN_NAME))
        HelperDisplayGpuResults(DISPLAY_CTXCREATE, DCGM_CONTEXT_CREATE_INDEX, diagResults, gpuIndices);
    else
        HelperDisplayGpuResults(DISPLAY_MEMORY, DCGM_MEMORY_INDEX, diagResults, gpuIndices);

    /* Don't show the hardware diagnostic if it skipped */
    bool skipped = true;

    for (size_t i = 0; i < gpuIndices.size(); i++)
    {
        if (diagResults[gpuIndices[i]].results[DCGM_DIAGNOSTIC_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            skipped = false;
            break;
        }
    }

    if (skipped == false)
        HelperDisplayGpuResults(DIAGNOSTIC_PLUGIN_NAME, DCGM_DIAGNOSTIC_INDEX, diagResults, gpuIndices);

    HelperDisplayGpuResults(PULSE_TEST_PLUGIN_NAME, DCGM_PULSE_TEST_INDEX, diagResults, gpuIndices);
}

/*****************************************************************************/
void Diag::HelperDisplayIntegration(dcgmDiagResponsePerGpu_v5 *diagResults, const std::vector<unsigned int> &gpuIndices)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_INTEGRATION;

    HelperDisplayGpuResults(PCIE_PLUGIN_NAME, DCGM_PCI_INDEX, diagResults, gpuIndices);
}

/*****************************************************************************/
void Diag::HelperDisplayPerformance(dcgmDiagResponsePerGpu_v5 *diagResults, const std::vector<unsigned int> &gpuIndices)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_STRESS;

    HelperDisplayGpuResults(SMSTRESS_PLUGIN_NAME, DCGM_SM_STRESS_INDEX, diagResults, gpuIndices);
    HelperDisplayGpuResults(TS_PLUGIN_NAME, DCGM_TARGETED_STRESS_INDEX, diagResults, gpuIndices);
    HelperDisplayGpuResults(TP_PLUGIN_NAME, DCGM_TARGETED_POWER_INDEX, diagResults, gpuIndices);
    HelperDisplayGpuResults(MEMBW_PLUGIN_NAME, DCGM_MEMORY_BANDWIDTH_INDEX, diagResults, gpuIndices);
    HelperDisplayGpuResults(MEMTEST_PLUGIN_NAME, DCGM_MEMTEST_INDEX, diagResults, gpuIndices);
    HelperDisplayGpuResults(EUD_PLUGIN_NAME, DCGM_EUD_TEST_INDEX, diagResults, gpuIndices);
}

/*****************************************************************************/
std::string Diag::HelperDisplayDiagResult(dcgmDiagResult_t val)
{
    if (val == DCGM_DIAG_RESULT_PASS)
    {
        return "Pass";
    }

    if (val == DCGM_DIAG_RESULT_SKIP)
    {
        return "Skip";
    }
    return "Fail";
}

/*****************************************************************************/
bool Diag::isWhitespace(char c)
{
    bool whitespace = false;

    switch (c)
    {
        case ' ':
        case '\n':
        case '\t':
        case '\r':
        case '\f':

            whitespace = true;
            break;
    }

    return (whitespace);
}

/*****************************************************************************/
std::string Diag::Sanitize(const std::string &toOutput)
{
    std::string sanitized;
    size_t pos;

    // Remove '***' and everything before it, if present
    if ((pos = toOutput.find("***")) != std::string::npos)
    {
        sanitized = toOutput.substr(pos + 3); // skip the "***"
    }
    else
    {
        sanitized = toOutput;
    }

    // Remove trailing whitespace
    while ((sanitized.size() > 0) && (isWhitespace(sanitized.at(sanitized.size() - 1))))
    {
        sanitized.erase(sanitized.size() - 1);
    }

    pos = 0;
    while ((sanitized.size() > 0) && (isWhitespace(sanitized.at(pos))))
    {
        pos++;
    }

    return (sanitized.substr(pos));
}

/*****************************************************************************/
void Diag::DisplayVerboseInfo(CommandOutputController &cmdView, const std::string &name, const std::string &info)
{
    // It can only display 45 characters at a time, so split larger messages onto different lines
    for (size_t pos = 0; pos < info.size(); pos += DATA_INFO_TAG_LEN)
    {
        // Only write the name for the first line
        if (pos == 0)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, name);
        }
        else
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, "");
        }
        cmdView.addDisplayParameter(DATA_INFO_TAG, info.substr(pos, DATA_INFO_TAG_LEN));
        cmdView.display();
    }
}

void Diag::HelperDisplayDetails(bool forceVerbose,
                                const std::vector<unsigned int> &gpuIndices,
                                unsigned int testIndex,
                                CommandOutputController &cmdView,
                                dcgmDiagResponsePerGpu_v5 *diagResults)
{
    bool displayInfo     = forceVerbose;
    bool displayWarnings = forceVerbose;

    if (m_drd.flags & DCGM_RUN_FLAGS_VERBOSE)
    {
        displayInfo     = true;
        displayWarnings = true;
    }

    if (displayWarnings)
    {
        for (unsigned int i = 0; i < gpuIndices.size(); i++)
        {
            unsigned int gpuIndex = gpuIndices[i];
            for (unsigned int j = 0; j < DCGM_MAX_ERRORS; j++)
            {
                if (diagResults[gpuIndex].results[testIndex].error[j].msg[0] != '\0')
                {
                    DisplayVerboseInfo(
                        cmdView, "Warning", Sanitize(diagResults[gpuIndex].results[testIndex].error[j].msg));
                }
            }
        }
    }

    if (displayInfo)
    {
        for (unsigned int i = 0; i < gpuIndices.size(); i++)
        {
            unsigned int gpuIndex = gpuIndices[i];
            if (diagResults[gpuIndex].results[testIndex].info[0] != '\0')
            {
                DisplayVerboseInfo(cmdView, "Info", Sanitize(diagResults[gpuIndex].results[testIndex].info));
            }
        }
    }
}


/*****************************************************************************/
void Diag::HelperDisplayGpuResults(std::string dataName,
                                   unsigned int testIndex,
                                   dcgmDiagResponsePerGpu_v5 *diagResults,
                                   const std::vector<unsigned int> &gpuIndices)
{
    CommandOutputController cmdView = CommandOutputController();
    std::stringstream ss;
    std::list<unsigned int> passed;
    std::list<unsigned int> failed;
    std::list<unsigned int> skipped;
    std::list<unsigned int> warned;

    bool isDisplayedFirst   = true;
    bool showWarnings       = false;
    size_t numGpus          = gpuIndices.size();
    std::string displayName = HelperGetPluginName(testIndex);

    cmdView.setDisplayStencil(DIAG_DATA);

    if (testIndex == DCGM_CONTEXT_CREATE_INDEX)
    {
        testIndex = 0; // Context create is only ever run by itself, and it's stored in index 0
    }

    for (unsigned int i = 0; i < numGpus; i++)
    {
        unsigned int gpuIndex = gpuIndices[i];

        if (diagResults[gpuIndex].results[testIndex].status == DCGM_DIAG_RESULT_PASS)
        {
            passed.push_back(diagResults[gpuIndex].gpuId);
        }
        else if (diagResults[gpuIndex].results[testIndex].status == DCGM_DIAG_RESULT_SKIP)
        {
            skipped.push_back(diagResults[gpuIndex].gpuId);
        }
        else if (diagResults[gpuIndex].results[testIndex].status == DCGM_DIAG_RESULT_WARN)
        {
            warned.push_back(diagResults[gpuIndex].gpuId);
        }
        else if (diagResults[gpuIndex].results[testIndex].status == DCGM_DIAG_RESULT_FAIL)
        {
            failed.push_back(diagResults[gpuIndex].gpuId);
        }
    }

    if (passed.size() == numGpus)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, displayName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Pass - All");
        cmdView.display();
        HelperDisplayDetails(false, gpuIndices, testIndex, cmdView, diagResults);
        return;
    }

    if (skipped.size() == numGpus)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, displayName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Skip - All");
        cmdView.display();
        HelperDisplayDetails(true, gpuIndices, testIndex, cmdView, diagResults);
        return;
    }

    if (failed.size() == numGpus)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, displayName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Fail - All");
        cmdView.display();
        HelperDisplayDetails(true, gpuIndices, testIndex, cmdView, diagResults);
        return;
    }

    if (warned.size() == numGpus)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, displayName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Warn - All");
        cmdView.display();
        HelperDisplayDetails(true, gpuIndices, testIndex, cmdView, diagResults);
        // special case for the diagnostic case to show the return code
        if (dataName == "Diagnostic")
        {
            std::stringstream eudReturnCode;
            eudReturnCode << "  Code: (" << std::setfill('0') << std::setw(12) << diagResults[0].hwDiagnosticReturn
                          << ")";
            cmdView.addDisplayParameter(DATA_NAME_TAG, "");
            cmdView.addDisplayParameter(DATA_INFO_TAG, eudReturnCode.str());
            cmdView.display();
        }
        return;
    }

    if (passed.size() > 0)
    {
        ss << "Pass - GPU" << ((passed.size() == 1) ? ": " : "s: ");
        for (std::list<unsigned int>::iterator it = passed.begin(); it != passed.end(); it++)
        {
            ss << *it << ((it == --passed.end()) ? "   " : ", ");
        }
        cmdView.addDisplayParameter(DATA_NAME_TAG, displayName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
        cmdView.display();
        isDisplayedFirst = false;
    }

    if (failed.size() > 0)
    {
        ss.str("");
        ss << "Fail - GPU" << ((failed.size() == 1) ? ": " : "s: ");
        for (std::list<unsigned int>::iterator it = failed.begin(); it != failed.end(); it++)
        {
            ss << *it << ((it == --failed.end()) ? "   " : ", ");
        }
        cmdView.addDisplayParameter(DATA_NAME_TAG, (isDisplayedFirst ? displayName : ""));
        cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
        cmdView.display();
        isDisplayedFirst = false;
        showWarnings     = true;
    }

    if (warned.size() > 0)
    {
        ss.str("");
        ss << "Warn - GPU" << ((warned.size() == 1) ? ": " : "s: ");
        for (std::list<unsigned int>::iterator it = warned.begin(); it != warned.end(); it++)
        {
            ss << *it << ((it == --warned.end()) ? "   " : ", ");
        }
        cmdView.addDisplayParameter(DATA_NAME_TAG, (isDisplayedFirst ? displayName : ""));
        cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
        cmdView.display();
        isDisplayedFirst = false;
        showWarnings     = true;
    }


    if (skipped.size() > 0)
    {
        ss.str("");
        ss << "Skip - GPU" << ((skipped.size() == 1) ? ": " : "s: ");
        for (std::list<unsigned int>::iterator it = skipped.begin(); it != skipped.end(); it++)
        {
            ss << *it << ((it == --skipped.end()) ? "   " : ", ");
        }
        cmdView.addDisplayParameter(DATA_NAME_TAG, (isDisplayedFirst ? displayName : ""));
        cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
        cmdView.display();
        showWarnings = true;
    }

    HelperDisplayDetails(showWarnings, gpuIndices, testIndex, cmdView, diagResults);

    return;
}

/*****************************************************************************/
void Diag::HelperJsonAddBasicTests(Json::Value &output, int &categoryIndex, dcgmDiagResponse_t &diagResult)
{
    Json::Value category;
    category[NVVS_HEADER] = DISPLAY_DEPLOYMENT;
    // Since the CUDA_RUNTIME_LIBRARY check is no longer valid, we don't want to increment the JSON array
    // for that test. All JSON array indexes after that test will be offset by -1 to avoid having a NULL
    // entry in the JSON array.
    int adjustment = 0;

    // Make the categories array and add each entry
    for (unsigned int testIndex = 0; testIndex < diagResult.levelOneTestCount; testIndex++)
    {
        Json::Value resultEntry;

        // Skip the Cuda Runtime library test when it is not run, which is always for now.
        if (testIndex == DCGM_SWTEST_CUDA_RUNTIME_LIBRARY
            && diagResult.levelOneResults[testIndex].status == DCGM_DIAG_RESULT_NOT_RUN)
        {
            adjustment = -1;
            continue;
        }

        category[NVVS_TESTS][testIndex + adjustment][NVVS_TEST_NAME] = levelOneTests[testIndex];
        resultEntry[NVVS_STATUS] = HelperDisplayDiagResult(diagResult.levelOneResults[testIndex].status);

        for (unsigned int i = 0; i < DCGM_MAX_ERRORS; i++)
        {
            Json::Value warningEntry;

            if (diagResult.levelOneResults[testIndex].error[i].msg[0] != '\0')
            {
                resultEntry[NVVS_GPU_ID]          = diagResult.levelOneResults[testIndex].error[i].gpuId;
                warningEntry[NVVS_WARNINGS]       = diagResult.levelOneResults[testIndex].error[i].msg;
                warningEntry[NVVS_ERROR_ID]       = diagResult.levelOneResults[testIndex].error[i].code;
                warningEntry[NVVS_ERROR_CATEGORY] = diagResult.levelOneResults[testIndex].error[i].category;
                warningEntry[NVVS_ERROR_SEVERITY] = diagResult.levelOneResults[testIndex].error[i].severity;

                resultEntry[NVVS_WARNINGS][i] = warningEntry;
            }
        }

        category[NVVS_TESTS][testIndex + adjustment][NVVS_RESULTS][0] = resultEntry;
    }

    // Add the categories array to the root json node
    output[NVVS_NAME][NVVS_HEADERS][categoryIndex] = category;
    categoryIndex++;
}

/*****************************************************************************/
/*
 * Returns the plugin name associated with the given index, or "" if not found
 */
std::string Diag::HelperGetPluginName(unsigned int index)
{
    switch (index)
    {
        case DCGM_MEMORY_INDEX:
        {
            if (!strcasecmp(m_drd.testNames[0], CTXCREATE_PLUGIN_NAME))
                return DISPLAY_CTXCREATE;
            else
                return DISPLAY_MEMORY;
        }

        case DCGM_DIAGNOSTIC_INDEX:
            return DISPLAY_DIAGNOSTIC;

        case DCGM_PCI_INDEX:
            return DISPLAY_PCIE;

        case DCGM_SM_STRESS_INDEX:
            return DISPLAY_SM_STRESS;

        case DCGM_TARGETED_STRESS_INDEX:
            return DISPLAY_TS;

        case DCGM_TARGETED_POWER_INDEX:
            return DISPLAY_TP;

        case DCGM_MEMORY_BANDWIDTH_INDEX:
            return DISPLAY_MEMBW;

        case DCGM_MEMTEST_INDEX:
            return DISPLAY_MEMTEST;

        case DCGM_PULSE_TEST_INDEX:
            return DISPLAY_PULSE_TEST;

        case DCGM_EUD_TEST_INDEX:
            return DISPLAY_EUD_TEST;

        case DCGM_CONTEXT_CREATE_INDEX:
            return DISPLAY_CTXCREATE;
    }

    return "";
}

/*****************************************************************************/
/*
 * Adds the result to this test entry and returns true, or returns false if this gpu didn't run the test.
 */
bool Diag::HelperJsonAddResult(dcgmDiagResponsePerGpu_v5 &gpuResult,
                               Json::Value &testEntry,
                               unsigned int gpuIndex,
                               unsigned int testIndex,
                               size_t i)
{
    Json::Value resultEntry;
    char buf[10];

    // Don't record an entry for tests that weren't run
    if (gpuResult.results[testIndex].status == DCGM_DIAG_RESULT_NOT_RUN)
    {
        return false;
    }

    snprintf(buf, sizeof(buf), "%u", gpuIndex);

    resultEntry[NVVS_GPU_ID] = buf;
    resultEntry[NVVS_STATUS] = HelperDisplayDiagResult(gpuResult.results[testIndex].status);

    for (unsigned int j = 0; j < DCGM_MAX_ERRORS; j++)
    {
        Json::Value warningEntry;

        if (gpuResult.results[testIndex].error[j].msg[0] != '\0')
        {
            warningEntry[NVVS_WARNING]        = gpuResult.results[testIndex].error[j].msg;
            warningEntry[NVVS_ERROR_ID]       = gpuResult.results[testIndex].error[j].code;
            warningEntry[NVVS_ERROR_CATEGORY] = gpuResult.results[testIndex].error[j].category;
            warningEntry[NVVS_ERROR_SEVERITY] = gpuResult.results[testIndex].error[j].severity;

            resultEntry[NVVS_WARNINGS][j] = warningEntry;
        }
    }

    if (gpuResult.results[testIndex].info[0] != '\0')
        resultEntry[NVVS_INFO] = gpuResult.results[testIndex].info;

    testEntry[NVVS_RESULTS][static_cast<int>(i)] = resultEntry;

    return true;
}

/*****************************************************************************/
/*
 * Adds the plugin output (represented by testEntry) to the category
 */
void Diag::HelperJsonAddPlugin(Json::Value &category, int &pluginCount, Json::Value &testEntry)
{
    category[NVVS_TESTS][pluginCount] = testEntry;
    pluginCount++;
}

/*****************************************************************************/
/*
 * Adds the category output to the category array
 */
void Diag::HelperJsonAddCategory(Json::Value &output, int &categoryIndex, Json::Value &category, int categoryCount)
{
    if (categoryCount > 0)
    {
        output[NVVS_NAME][NVVS_HEADERS][categoryIndex] = category;
        categoryIndex++;
    }
}

/*****************************************************************************/
/*
 * Builds the json based on the contents of diagResult
 */
void Diag::HelperJsonBuildOutput(Json::Value &output,
                                 dcgmDiagResponse_t &diagResult,
                                 const std::vector<unsigned int> &gpuIndices)
{
    int categoryIndex = 0;

    Json::Value hardware;
    Json::Value integration;
    Json::Value stress;
    int hardwarePluginCount    = 0;
    int integrationPluginCount = 0;
    int stressPluginCount      = 0;
    hardware[NVVS_HEADER]      = DISPLAY_HARDWARE;
    integration[NVVS_HEADER]   = DISPLAY_INTEGRATION;
    stress[NVVS_HEADER]        = DISPLAY_STRESS;

    std::string gpuList = m_drd.gpuList;

    // Make sure we have an accurate gpu list
    if (gpuList.size() == 0)
    {
        std::stringstream buf;
        for (size_t i = 0; i < gpuIndices.size(); i++)
        {
            if (i > 0)
                buf << "," << i;
            else
                buf << i;
        }
    }

    output[NVVS_VERSION_STR]    = diagResult.dcgmVersion;
    output[NVVS_DRIVER_VERSION] = diagResult.driverVersion;

    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        if (strcmp(diagResult.devSerials[i], DCGM_STR_BLANK))
        {
            output[NVVS_GPU_SERIALS][std::to_string(i)] = diagResult.devSerials[i];
        }
    }

    for (unsigned int i = 0; i < diagResult.gpuCount; i++)
    {
        output[NVVS_GPU_DEV_IDS][i] = diagResult.devIds[i];
    }

    HelperJsonAddBasicTests(output, categoryIndex, diagResult);

    // Now get each of the other test's results
    for (unsigned int pluginIndex = 0; pluginIndex < DCGM_PER_GPU_TEST_COUNT_V8; pluginIndex++)
    {
        Json::Value testEntry;
        std::string testName      = HelperGetPluginName(pluginIndex);
        testEntry[NVVS_TEST_NAME] = testName;
        bool pluginRan            = false;

        for (size_t i = 0; i < gpuIndices.size(); i++)
        {
            unsigned int gpuIndex = gpuIndices[i];
            if (HelperJsonAddResult(diagResult.perGpuResponses[gpuIndex], testEntry, gpuIndex, pluginIndex, i) == true)
                pluginRan = true;
        }

        if (pluginRan)
        {
            switch (pluginIndex)
            {
                case DCGM_MEMORY_INDEX:
                case DCGM_DIAGNOSTIC_INDEX:
                case DCGM_PULSE_TEST_INDEX:
                case DCGM_EUD_TEST_INDEX:

                    HelperJsonAddPlugin(hardware, hardwarePluginCount, testEntry);
                    break;

                case DCGM_PCI_INDEX:

                    HelperJsonAddPlugin(integration, integrationPluginCount, testEntry);
                    break;

                case DCGM_SM_STRESS_INDEX:
                case DCGM_TARGETED_STRESS_INDEX:
                case DCGM_TARGETED_POWER_INDEX:
                case DCGM_MEMORY_BANDWIDTH_INDEX:
                case DCGM_MEMTEST_INDEX:

                    HelperJsonAddPlugin(stress, stressPluginCount, testEntry);
                    break;
            }
        }
    }

    HelperJsonAddCategory(output, categoryIndex, integration, integrationPluginCount);
    HelperJsonAddCategory(output, categoryIndex, hardware, hardwarePluginCount);
    HelperJsonAddCategory(output, categoryIndex, stress, stressPluginCount);
}

/*****************************************************************************/
/*
 * Displays diagResult as json instead of the normal output
 */
dcgmReturn_t Diag::HelperDisplayAsJson(dcgmDiagResponse_t &diagResult, const std::vector<unsigned int> &gpuIndices)
{
    Json::Value output;
    dcgmReturn_t result = DCGM_ST_OK;

    HelperJsonBuildOutput(output, diagResult, gpuIndices);

    if (m_iterations <= 1)
    {
        std::cout << output.toStyledString();
    }
    else
    {
        m_jsonTmpValue = output;
    }

    return result;
}

/*****************************************************************************
 *****************************************************************************
 * Start Diagnostics Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
StartDiag::StartDiag(const std::string &hostname,
                     const bool hostAddressWasOverridden,
                     const std::string &parms,
                     const std::string &configPath,
                     bool jsonOutput,
                     dcgmRunDiag_t &drd,
                     unsigned int iterations,
                     const std::string &pathToDcgmExecutable)
    : m_diagObj(iterations, hostname)

{
    std::string configFileContents;
    drd.version = dcgmRunDiag_version;
    m_hostName  = hostname;

    /* If the host address was overridden, complain if we can't connect.
       Otherwise, don't complain and just start an embedded host engine instead */
    m_silent = !hostAddressWasOverridden;

    // Parms is in the format: test_name.attr_name=attr_value[;...]
    // Parse it
    if (parms.size() > 0)
    {
        std::vector<std::string> parmsVec;
        dcgmTokenizeString(parms, ";", parmsVec);

        // Make sure each parameter is properly formatted
        for (size_t i = 0; i < parmsVec.size(); i++)
        {
            if (parmsVec[i].find('=') == std::string::npos)
            {
                std::string err_text("Improperly formatted parameters argument: '");
                err_text += parms + "'. Argument must follow the format: test_name.attr_name=attr_value[;...]";

                throw TCLAP::CmdLineParseException(err_text);
            }
        }
    }
    else if (configPath.size() > 0)
    {
        std::ifstream configFile;
        std::stringstream ss;

        if (filesize(configPath) > DCGM_MAX_CONFIG_FILE_LEN)
        {
            std::string err_text;
            ss << "Config file too large. Its size (" << filesize(configPath) << ") exceeds "
               << DCGM_MAX_CONFIG_FILE_LEN;
            throw TCLAP::CmdLineParseException(ss.str());
        }

        configFile.open(configPath.c_str());

        if (configFile.good() == false)
        {
            std::string err_text("Could not open configuration file: '");
            err_text += configPath + "'";
            throw TCLAP::CmdLineParseException(err_text);
        }

        ss.clear();
        ss << configFile.rdbuf();
        dcgm_diag_common_set_config_file_contents(ss.str(), drd);
    }
    // Check for valid gpu list format
    std::string gpuList(drd.gpuList);
    if (validGpuListFormat(gpuList) == false)
    {
        std::string err_text("Gpu list '");
        err_text += gpuList + "' must be a comma-separated list of numbers";
        throw TCLAP::CmdLineParseException(err_text);
    }

    this->m_diagObj.setDcgmRunDiag(&drd);
    this->m_diagObj.setJsonOutput(jsonOutput);

    // Set path to dcgm executable. This is used by the signal handler to stop the launched diagnostic if needed.
    diag_pathToExecutable = pathToDcgmExecutable;
}

/*****************************************************************************/
bool StartDiag::validGpuListFormat(const std::string &gpuList)
{
    bool valid = true;

    std::vector<std::string> gpuIndices;
    dcgmTokenizeString(gpuList, ",", gpuIndices);

    for (size_t i = 0; i < gpuIndices.size(); i++)
    {
        if (isdigit(gpuIndices[i].c_str()[0]) == false)
        {
            valid = false;
            break;
        }
    }

    return (valid);
}

/*****************************************************************************/
dcgmReturn_t StartDiag::StartListenerServer()
{
    dcgmReturn_t ret          = DCGM_ST_OK;
    unsigned short listenPort = 5555;
    char listenIp[32];
    sprintf(listenIp, "127.0.0.1");
    int connTcp = 1;

    ret = dcgmEngineRun(listenPort, listenIp, connTcp);

    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Err: Can't listen for incoming queries, so DCGM Diag can't get telemetry: %d.\n", ret);
        return ret;
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t StartDiag::DoExecuteConnected()
{
    m_silent = true;

    // Set global hostname so that the signal handler can terminate a launched diagnostic if necessary
    diag_hostname = m_hostName;

    dcgmReturn_t ret = m_diagObj.RunStartDiag(m_dcgmHandle);

    // reset global hostname
    diag_hostname = "";

    return ret;
}

dcgmReturn_t StartDiag::DoExecuteConnectionFailure(dcgmReturn_t connectionStatus)
{
    m_silent = true;

    // Set global hostname so that the signal handler can terminate a launched diagnostic if necessary
    diag_hostname = m_hostName;

    // Attempt to start an embedded host engine
    dcgmStartEmbeddedV2Params_v1 params {};
    params.version  = dcgmStartEmbeddedV2Params_version1;
    params.opMode   = DCGM_OPERATION_MODE_AUTO;
    params.logFile  = nullptr;
    params.severity = DcgmLoggingSeverityUnspecified;

    connectionStatus = dcgmStartEmbedded_v2(&params);

    if (connectionStatus != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to start an embedded host engine. " << errorString(connectionStatus) << "."
                  << std::endl;
    }
    else
    {
        connectionStatus = m_diagObj.RunStartDiag(params.dcgmHandle);

        dcgmStopEmbedded(params.dcgmHandle);
    }

    // reset global hostname
    diag_hostname = "";

    return connectionStatus;
}


/*****************************************************************************
 *****************************************************************************
 * Abort Diagnostics Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
AbortDiag::AbortDiag(std::string hostname)
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t AbortDiag::DoExecuteConnected()
{
    return dcgmStopDiagnostic(m_dcgmHandle);
}
