#!/usr/bin/env bash
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -ex

read -r _ URL SHA512SUM <<<$(grep '^git-lfs ' $1)

curl --location --fail --output git-lfs.tar.gz $URL
echo "$SHA512SUM git-lfs.tar.gz" | sha512sum --check -

mkdir git-lfs
tar xf git-lfs.tar.gz -C git-lfs
git-lfs/install.sh

rm -rf git-lfs.tar.gz git-lfs
