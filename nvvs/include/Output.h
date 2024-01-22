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
#ifndef NVVS_NVVS_Output_H
#define NVVS_NVVS_Output_H

#include "NvvsCommon.h"
#include "PluginInterface.h"
#include <any>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

class Output
{
    /***************************PUBLIC***********************************/
public:
    Output()
        : m_out(std::cout.rdbuf())
        , m_err(std::cerr.rdbuf())
    {
        if (nvvsCommon.quietMode)
        {
            m_out.rdbuf(m_nullSS.rdbuf());
            m_err.rdbuf(m_nullSS.rdbuf());
        }
    }

    virtual ~Output() = default;

    enum fillType_enum
    {
        NVVS_FILL_PREFACE,
        NVVS_FILL_DELIMITER1,
        NVVS_FILL_DELIMITER2,
        NVVS_FILL_DOT,
    };

    virtual void header(const std::string &headerSting);
    virtual void Result(nvvsPluginResult_t overallResult,
                        const std::vector<dcgmDiagSimpleResult_t> &perGpuResults,
                        const std::vector<dcgmDiagErrorDetail_v2> &errors,
                        const std::vector<dcgmDiagErrorDetail_v2> &info,
                        const std::optional<std::any> &pluginSpecificData = std::nullopt);
    virtual void prep(const std::string &testString);
    virtual void updatePluginProgress(unsigned int progress, bool clear);
    virtual void print();
    virtual void addInfoStatement(const std::string &info);
    virtual void AddGpusAndDriverVersion(std::vector<Gpu *> &gpuList);

    /***************************PRIVATE**********************************/
private:
    std::stringstream m_nullSS;

    // methods
    std::string fill(fillType_enum type);

protected:
    std::ostream m_out;
    std::ostream m_err;

    std::string resultEnumToString(nvvsPluginResult_t result);
    std::vector<std::string> globalInfo;

    std::string RemoveNewlines(const std::string &str);

    void WriteInfo(const std::vector<dcgmDiagErrorDetail_v2> &info);
};

#endif // NVVS_NVVS_Output_H
