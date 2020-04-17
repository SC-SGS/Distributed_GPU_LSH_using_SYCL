/******************************************************************************
 *
 * Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC
 * and other gtest-mpi-listener developers. See the COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
*******************************************************************************/
/*******************************************************************************
 * An example from Google Test was copied with minor modifications. The
 * license of Google Test is below.
 *
 * Google Test has the following copyright notice, which must be
 * duplicated in its entirety per the terms of its license:
 *
 *  Copyright 2005, Google Inc.  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are
 *  met:
 *
 *      * Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following disclaimer
 *  in the documentation and/or other materials provided with the
 *  distribution.
 *      * Neither the name of Google Inc. nor the names of its
 *  contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#ifndef GTEST_MPI_MINIMAL_LISTENER_H
#define GTEST_MPI_MINIMAL_LISTENER_H

#include <cassert>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>


namespace GTestMPIListener {

    // This class sets up the global test environment, which is needed to finalize MPI.
    class MPIEnvironment : public ::testing::Environment {
        public:
            MPIEnvironment() : ::testing::Environment() { }
            MPIEnvironment(const MPIEnvironment&) = delete;

            virtual ~MPIEnvironment() = default;

            virtual void SetUp() override {
                int is_mpi_initialized;
                ASSERT_EQ(MPI_Initialized(&is_mpi_initialized), MPI_SUCCESS);
                if (!is_mpi_initialized) {
                    std::cout << "MPI must be initialized before RUN_ALL_TESTS!\n"
                              << "Add '::testing::InitGoogleTest(&argc, argv);\n"
                              << "     MPI_Init(&argc, &argv);' to your 'main' function!" << std::endl;
                    FAIL();
                }
            }

            virtual void TearDown() override {
                int is_mpi_finalized;
                ASSERT_EQ(MPI_Finalized(&is_mpi_finalized), MPI_SUCCESS);
                if (!is_mpi_finalized) {
                    int rank;
                    ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
                    ASSERT_EQ(MPI_Finalize(), MPI_SUCCESS);
                }
                ASSERT_EQ(MPI_Finalized(&is_mpi_finalized), MPI_SUCCESS);
                ASSERT_TRUE(is_mpi_finalized);
            }
    };


    // This class more or less takes the code in Google Test's
    // MinimalistPrinter example and wraps certain parts of it in MPI calls,
    // gathering all results onto rank zero.
    class MPIMinimalistPrinter : public ::testing::EmptyTestEventListener {
        public:
            explicit MPIMinimalistPrinter(MPI_Comm comm = MPI_COMM_WORLD) : ::testing::EmptyTestEventListener(), result_vector_() {
                int is_mpi_initialized;
                assert(MPI_Initialized(&is_mpi_initialized) == MPI_SUCCESS);
                if (!is_mpi_initialized) {
                    std::cout << "MPI must be initialized before RUN_ALL_TESTS!\n"
                              << "Add '::testing::InitGoogleTest(&argc, argv);\n"
                              << "     MPI_Init(&argc, &argv);' to your 'main' function!" << std::endl;
                    assert(false);
                }
                MPI_Comm_dup(comm, &comm_);
                UpdateCommState();
            }

            MPIMinimalistPrinter(const MPIMinimalistPrinter& other) : result_vector_(other.result_vector_) {
                MPI_Comm_dup(other.comm_, &comm_);
                UpdateCommState();
            }

            // Called before the Environment is torn down.
            void OnEnvironmentTearDownStart() {
                int is_mpi_finalized;
                assert(MPI_Finalized(&is_mpi_finalized) == MPI_SUCCESS);
                if (!is_mpi_finalized) {
                    MPI_Comm_free(&comm_);
                }
            }

            // Called before a test starts.
            virtual void OnTestStart(const ::testing::TestInfo& test_info) override {
                // Only need to report test start info on rank 0
                if (rank_ == 0) {
                    std::printf("*** Test %s.%s starting.\n", test_info.test_case_name(), test_info.name());
                }
            }

            // Called after an assertion failure or an explicit SUCCESS() macro.
            // In an MPI program, this means that certain ranks may not call this
            // function if a test part does not fail on all ranks. Consequently, it
            // is difficult to have explicit synchronization points here.
            virtual void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
                result_vector_.push_back(test_part_result);
            }

            // Called after a test ends.
            virtual void OnTestEnd(const ::testing::TestInfo& test_info) override {
                int localResultCount = result_vector_.size();
                std::vector<int> resultCountOnRank(size_, 0);
                MPI_Gather(&localResultCount, 1, MPI_INT, resultCountOnRank.data(), 1, MPI_INT, 0, comm_);

                if (rank_ != 0) {
                    // Nonzero ranks send constituent parts of each result to rank 0
                    for (const ::testing::TestPartResult test_part_result : result_vector_) {
                        int resultStatus = test_part_result.failed();
                        std::string resultFileName(test_part_result.file_name());
                        int resultLineNumber = test_part_result.line_number();
                        std::string resultSummary(test_part_result.summary());

                        // Must add one for null termination
                        int resultFileNameSize = resultFileName.size() + 1;
                        int resultSummarySize = resultSummary.size() + 1;

                        MPI_Send(&resultStatus, 1, MPI_INT, 0, rank_, comm_);
                        MPI_Send(&resultFileNameSize, 1, MPI_INT, 0, rank_, comm_);
                        MPI_Send(&resultLineNumber, 1, MPI_INT, 0, rank_, comm_);
                        MPI_Send(&resultSummarySize, 1, MPI_INT, 0, rank_, comm_);
                        MPI_Send(resultFileName.c_str(), resultFileNameSize, MPI_CHAR, 0, rank_, comm_);
                        MPI_Send(resultSummary.c_str(), resultSummarySize, MPI_CHAR, 0, rank_, comm_);
                    }
                } else {
                    // Rank 0 first prints its local result data
                    for (const ::testing::TestPartResult test_part_result : result_vector_) {
                        std::printf("      %s on rank %i, %s:%i\n%s\n",
                                    test_part_result.failed() ? "*** Failure" : "Success",
                                    rank_,
                                    test_part_result.file_name(),
                                    test_part_result.line_number(),
                                    test_part_result.summary());
                    }

                    for (int r = 1; r < size_; ++r) {
                        for (int i = 0; i < resultCountOnRank[r]; ++i) {
                            int resultStatus, resultFileNameSize, resultLineNumber, resultSummarySize;
                            MPI_Recv(&resultStatus, 1, MPI_INT, r, r, comm_, MPI_STATUS_IGNORE);
                            MPI_Recv(&resultFileNameSize, 1, MPI_INT, r, r, comm_, MPI_STATUS_IGNORE);
                            MPI_Recv(&resultLineNumber, 1, MPI_INT, r, r, comm_, MPI_STATUS_IGNORE);
                            MPI_Recv(&resultSummarySize, 1, MPI_INT, r, r, comm_, MPI_STATUS_IGNORE);

                            std::string resultFileName(resultFileNameSize, ' ');
                            std::string resultSummary(resultSummarySize, ' ');
                            MPI_Recv(resultFileName.data(), resultFileNameSize, MPI_CHAR, r, r, comm_, MPI_STATUS_IGNORE);
                            MPI_Recv(resultSummary.data(), resultSummarySize, MPI_CHAR, r, r, comm_, MPI_STATUS_IGNORE);

                            std::printf("      %s on rank %i, %s:%i\n%s\n",
                                    resultStatus ? "*** Failure" : "Success",
                                    r,
                                    resultFileName.c_str(),
                                    resultLineNumber,
                                    resultSummary.c_str());
                        }
                    }

                    std::printf("*** Test %s.%s ending.\n", test_info.test_case_name(), test_info.name());
                }

                result_vector_.clear();
            }

        private:
            MPI_Comm comm_;
            int rank_;
            int size_;
            std::vector<::testing::TestPartResult> result_vector_;

            int UpdateCommState() {
                int flag = MPI_Comm_rank(comm_, &rank_);
                if (flag != MPI_SUCCESS) { return flag; }
                flag = MPI_Comm_size(comm_, &size_);
                return flag;
            }
    };


    // This class more or less takes the code in Google Test's
    // MinimalistPrinter example and wraps certain parts of it in MPI calls,
    // gathering all results onto rank zero.
    class MPIWrapperPrinter : public ::testing::TestEventListener {
        public:
            MPIWrapperPrinter(::testing::TestEventListener* l, MPI_Comm comm)
                    : ::testing::TestEventListener(), listener_(l), result_vector_() {
                int is_mpi_initialized = 0;
                assert(MPI_Initialized(&is_mpi_initialized) == MPI_SUCCESS);
                if (!is_mpi_initialized) {
                    std::cout << "MPI must be initialized before RUN_ALL_TESTS!\n"
                              << "Add '::testing::InitGoogleTest(&argc, argv);\n"
                              << "     MPI_Init(&argc, &argv);' to your 'main' function!" << std::endl;
                    assert(false);
                }
                MPI_Comm_dup(comm, &comm_);
                UpdateCommState();
            }

            MPIWrapperPrinter(const MPIWrapperPrinter& other) : listener_(other.listener_), result_vector_(other.result_vector_) {
                MPI_Comm_dup(other.comm_, &comm_);
                UpdateCommState();
            }

            // Called before test activity starts
            virtual void OnTestProgramStart(const ::testing::UnitTest& unit_test) override {
                if (rank_ == 0) { listener_->OnTestProgramStart(unit_test); }
            }


            // Called before each test iteration starts, where iteration is
            // the iterate index. There could be more than one iteration if
            // GTEST_FLAG(repeat) is used.
            virtual void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override {
                if (rank_ == 0) { listener_->OnTestIterationStart(unit_test, iteration); }
            }



            // Called before environment setup before start of each test iteration
            virtual void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) override {
                if (rank_ == 0) { listener_->OnEnvironmentsSetUpStart(unit_test); }
            }

            virtual void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test) override {
                if (rank_ == 0) { listener_->OnEnvironmentsSetUpEnd(unit_test); }
            }

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
            virtual void OnTestCaseStart(const ::testing::TestCase& test_case) override {
                if (rank_ == 0) { listener_->OnTestCaseStart(test_case); }
            }
#endif // GTEST_REMOVE_LEGACY_TEST_CASEAPI_

            // Called before a test starts.
            virtual void OnTestStart(const ::testing::TestInfo& test_info) override {
                // Only need to report test start info on rank 0
                if (rank_ == 0) { listener_->OnTestStart(test_info); }
            }

            // Called after an assertion failure or an explicit SUCCESS() macro.
            // In an MPI program, this means that certain ranks may not call this
            // function if a test part does not fail on all ranks. Consequently, it
            // is difficult to have explicit synchronization points here.
            virtual void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
                result_vector_.push_back(test_part_result);
                if (rank_ == 0) { listener_->OnTestPartResult(test_part_result); }
            }

            // Called after a test ends.
            virtual void OnTestEnd(const ::testing::TestInfo& test_info) override {
                int localResultCount = result_vector_.size();
                std::vector<int> resultCountOnRank(size_, 0);
                MPI_Gather(&localResultCount, 1, MPI_INT, resultCountOnRank.data(), 1, MPI_INT, 0, comm_);

                if (rank_ != 0) {
                    // Nonzero ranks send constituent parts of each result to rank 0
                    for (const ::testing::TestPartResult test_part_result : result_vector_) {
                        int resultStatus = test_part_result.failed();
                        std::string resultFileName(test_part_result.file_name());
                        int resultLineNumber = test_part_result.line_number();
                        std::string resultMessage(test_part_result.message());

                        int resultFileNameSize = resultFileName.size() + 1;
                        int resultMessageSize = resultMessage.size() + 1;

                        MPI_Send(&resultStatus, 1, MPI_INT, 0, rank_, comm_);
                        MPI_Send(&resultFileNameSize, 1, MPI_INT, 0, rank_, comm_);
                        MPI_Send(&resultLineNumber, 1, MPI_INT, 0, rank_, comm_);
                        MPI_Send(&resultMessageSize, 1, MPI_INT, 0, rank_, comm_);
                        MPI_Send(resultFileName.c_str(), resultFileNameSize, MPI_CHAR, 0, rank_, comm_);
                        MPI_Send(resultMessage.c_str(), resultMessageSize, MPI_CHAR, 0, rank_, comm_);
                    }
                } else {
                    // Rank 0 first prints its local result data
                    for (const ::testing::TestPartResult test_part_result : result_vector_) {
                        if (test_part_result.failed()) {
                            std::string message(test_part_result.message());
                            std::istringstream input_stream(message);
                            std::stringstream to_stream_into_failure;
                            std::string line_as_string;
                            while (std::getline(input_stream, line_as_string)) {
                                to_stream_into_failure << "[Rank 0 / " << size_ << "] " << line_as_string << std::endl;
                            }

                            ADD_FAILURE_AT(test_part_result.file_name(), test_part_result.line_number()) << to_stream_into_failure.str();
                        }
                    }

                    for (int r = 1; r < size_; r++) {
                        for (int i = 0; i < resultCountOnRank[r]; i++) {
                            int resultStatus, resultFileNameSize, resultLineNumber, resultMessageSize;
                            MPI_Recv(&resultStatus, 1, MPI_INT, r, r, comm_, MPI_STATUS_IGNORE);
                            MPI_Recv(&resultFileNameSize, 1, MPI_INT, r, r, comm_, MPI_STATUS_IGNORE);
                            MPI_Recv(&resultLineNumber, 1, MPI_INT, r, r, comm_, MPI_STATUS_IGNORE);
                            MPI_Recv(&resultMessageSize, 1, MPI_INT, r, r, comm_, MPI_STATUS_IGNORE);

                            std::string resultFileName(resultFileNameSize, ' ');
                            std::string resultMessage(resultMessageSize, ' ');
                            MPI_Recv(resultFileName.data(), resultFileNameSize, MPI_CHAR, r, r, comm_, MPI_STATUS_IGNORE);
                            MPI_Recv(resultMessage.data(), resultMessageSize, MPI_CHAR, r, r, comm_, MPI_STATUS_IGNORE);

                            if (resultStatus == 1) {
                                std::string message(resultMessage);
                                std::istringstream input_stream(message);
                                std::stringstream to_stream_into_failure;
                                std::string line_as_string;

                                while (std::getline(input_stream, line_as_string)) {
                                    to_stream_into_failure << "[Rank " << r << " / "  << size_ << "] " << line_as_string << std::endl;
                                }

                                ADD_FAILURE_AT(resultFileName.c_str(), resultLineNumber) << to_stream_into_failure.str();
                            }
                        }
                    }
                }

            result_vector_.clear();
            if (rank_ == 0) { listener_->OnTestEnd(test_info); }
        }

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
            virtual void OnTestCaseEnd(const ::testing::TestCase& test_case) override  {
                if (rank_ == 0) { listener_->OnTestCaseEnd(test_case); }
            }
#endif

            // Called before the Environment is torn down.
            virtual void OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) override {
                int is_mpi_finalized;
                ASSERT_EQ(MPI_Finalized(&is_mpi_finalized), MPI_SUCCESS);
                if (!is_mpi_finalized) {
                    MPI_Comm_free(&comm_);
                }
                if (rank_ == 0) { listener_->OnEnvironmentsTearDownStart(unit_test);  }
            }

            virtual void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) override {
                if (rank_ == 0) { listener_->OnEnvironmentsTearDownEnd(unit_test); }
            }

            virtual void OnTestIterationEnd(const ::testing::UnitTest &unit_test, int iteration) override {
                if (rank_ == 0) { listener_->OnTestIterationEnd(unit_test, iteration); }
            }

            // Called when test driver program ends
            virtual void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
                if (rank_ == 0) { listener_->OnTestProgramEnd(unit_test); }
            }


        private:
            // Use a pointer here instead of a reference because
            // ::testing::TestEventListeners::Release returns a pointer
            // (namely, one of type ::testing::TesteEventListener*).
            ::testing::TestEventListener* listener_;
            MPI_Comm comm_;
            int rank_;
            int size_;
            std::vector<::testing::TestPartResult> result_vector_;

            int UpdateCommState() {
                int flag = MPI_Comm_rank(comm_, &rank_);
                if (flag != MPI_SUCCESS) { return flag; }
                flag = MPI_Comm_size(comm_, &size_);
                return flag;
            }
    };

}

#endif // GTEST_MPI_MINIMAL_LISTENER_H