#ifndef SUFFIXARRAYPERFORMANCEMEASUREMENTS_H
#define SUFFIXARRAYPERFORMANCEMEASUREMENTS_H

#include "performancerecorder.hpp"
#include "stages.h"
#include <cuda_runtime.h>
#include "cuda_helpers.h"

class CudaMeasureHelper {
public:
    CudaMeasureHelper()
    {
        //(0);
        cudaEventCreate(&mstart);
        cudaEventCreate(&mstop);
        CUERR;
    }

    ~CudaMeasureHelper() {
        //(0);
        cudaEventDestroy(mstart);
        cudaEventDestroy(mstop);
        CUERR;
    }

    CudaMeasureHelper(const CudaMeasureHelper& other) = delete;

    void start() {
        //(0);
        cudaEventRecord(mstart, 0);
        CUERR;
    }

    float stop() {
        //(0);
        float elapsed;
        cudaEventRecord(mstop, 0);
        cudaEventSynchronize(mstop);
        cudaEventElapsedTime(&elapsed, mstart, mstop);
        CUERR;
        return elapsed;
    }

private:
    cudaEvent_t mstart, mstop;
};

class SuffixArrayPerformanceMeasurements
{
public:
    SuffixArrayPerformanceMeasurements(size_t max_repetitions) :
        mmain_performance_recorder(perf_rec::main_stage_names, 1),
        mloop_performance_recorder(perf_rec::loop_stage_names, max_repetitions),
        mwrite_isa_performance_recorder(perf_rec::write_isa_stage_names, max_repetitions),
        mfetch_rank_performance_recorder(perf_rec::fetch_rank_stage_names, max_repetitions),
        mprepare_final_merge_performance_recorder(perf_rec::prepare_final_merge_stages_names, 1)
    {
    }

    void start_main_stage(perf_rec::MainStages stage) {
        mmain_helpers[int(stage)].start();
    }

    void stop_main_stage(perf_rec::MainStages stage) {
        float t = mmain_helpers[int(stage)].stop();
        mmain_performance_recorder.register_measurement(stage, t);
    }

    void start_loop_stage(perf_rec::LoopStages stage) {
        mloop_helpers[int(stage)].start();
    }

    void stop_loop_stage(perf_rec::LoopStages stage) {
        float t = mloop_helpers[int(stage)].stop();
        mloop_performance_recorder.register_measurement(stage, t);
    }

    void start_write_isa_stage(perf_rec::WriteISAStages stage) {
        mwrite_isa_helpers[int(stage)].start();
    }

    void stop_write_isa_stage(perf_rec::WriteISAStages stage) {
        float t = mwrite_isa_helpers[int(stage)].stop();
        mwrite_isa_performance_recorder.register_measurement(stage, t);
    }

    void start_fetch_rank_stage(perf_rec::FetchRankStages stage) {
        mfetch_rank_helpers[int(stage)].start();
    }

    void stop_fetch_rank_stage(perf_rec::FetchRankStages stage) {
        float t = mfetch_rank_helpers[int(stage)].stop();
        mfetch_rank_performance_recorder.register_measurement(stage, t);
    }

    void start_prepare_final_merge_stage(perf_rec::PrepareFinalMergeStages stage) {
        mprepare_final_merge_helpers[int(stage)].start();
    }

    void stop_prepare_final_merge_stage(perf_rec::PrepareFinalMergeStages stage) {
        float t = mprepare_final_merge_helpers[int(stage)].stop();
        mprepare_final_merge_performance_recorder.register_measurement(stage, t);
    }

    void start_loop() {
        mwrite_isa_performance_recorder.next_iteration();
    }

    void next_iteration() {
        mloop_performance_recorder.next_iteration();
        mwrite_isa_performance_recorder.next_iteration();
        mfetch_rank_performance_recorder.next_iteration();
    }

    void done() {
        mmain_performance_recorder.done();
        mloop_performance_recorder.done();
        mwrite_isa_performance_recorder.done();
        mfetch_rank_performance_recorder.done();
    }

    void print(char* outputFile = "output") const {
        std::cout << "\nWrite ISA:\n";
        mwrite_isa_performance_recorder.print();
        std::cout << "\nFetch Rank:\n";
        mfetch_rank_performance_recorder.print();

        std::cout << "\nLoop:\n";
        // mloop_performance_recorder
        mloop_performance_recorder.print();
        mloop_performance_recorder.writeMeas(outputFile);
        std::cout << "\nPrepare final merge:\n";
        mprepare_final_merge_performance_recorder.print();

        std::cout << "\n\nMain:\n";
        mmain_performance_recorder.print();
        mmain_performance_recorder.writeMeas(outputFile);
        // mmain_performance_recorder.write();
        float total = mmain_performance_recorder.get_total() + mloop_performance_recorder.get_total();
        std::cout << "\n" << "Total: " << total << "\n\n";
    }

private:
    static const size_t NO_MAIN_STAGES = size_t(perf_rec::MainStages::NO_STAGES);
    static const size_t NO_LOOP_STAGES = size_t(perf_rec::LoopStages::NO_STAGES);
    static const size_t NO_WRITE_ISA_STAGES = size_t(perf_rec::WriteISAStages::NO_STAGES);
    static const size_t NO_FETCH_RANK_STAGES = size_t(perf_rec::FetchRankStages::NO_STAGES);
    static const size_t NO_PREPARE_FINAL_MERGE_STAGES = size_t(perf_rec::PrepareFinalMergeStages::NO_STAGES);

    perf_rec::PerformanceRecorder<perf_rec::MainStages, NO_MAIN_STAGES, float>
        mmain_performance_recorder;

    perf_rec::PerformanceRecorder<perf_rec::LoopStages, NO_LOOP_STAGES, float>
        mloop_performance_recorder;

    perf_rec::PerformanceRecorder<perf_rec::WriteISAStages, NO_WRITE_ISA_STAGES, float>
        mwrite_isa_performance_recorder;

    perf_rec::PerformanceRecorder<perf_rec::FetchRankStages, NO_FETCH_RANK_STAGES, float>
        mfetch_rank_performance_recorder;

    perf_rec::PerformanceRecorder<perf_rec::PrepareFinalMergeStages, NO_PREPARE_FINAL_MERGE_STAGES, float>
        mprepare_final_merge_performance_recorder;

    std::array<CudaMeasureHelper, NO_MAIN_STAGES> mmain_helpers;
    std::array<CudaMeasureHelper, NO_LOOP_STAGES> mloop_helpers;
    std::array<CudaMeasureHelper, NO_WRITE_ISA_STAGES> mwrite_isa_helpers;
    std::array<CudaMeasureHelper, NO_FETCH_RANK_STAGES> mfetch_rank_helpers;
    std::array<CudaMeasureHelper, NO_PREPARE_FINAL_MERGE_STAGES> mprepare_final_merge_helpers;
};

#endif // SUFFIXARRAYPERFORMANCEMEASUREMENTS_H
