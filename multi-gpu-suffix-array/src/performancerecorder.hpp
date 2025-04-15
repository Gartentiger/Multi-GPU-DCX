#ifndef PERFORMANCERECORDER_H
#define PERFORMANCERECORDER_H

#include <array>
#include <vector>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include "util.h"

namespace perf_rec {

template <typename STAGE_ENUM, size_t NO_STAGES, typename TIME_MEASURE_T>
class SimplePerformanceRecorder {
    public:
        static const size_t _NO_STAGES = NO_STAGES;
        using measurements_t = std::array<TIME_MEASURE_T, NO_STAGES>;

        SimplePerformanceRecorder()
        {
            clear();
        }

        void set(STAGE_ENUM stage, TIME_MEASURE_T time) {
            ASSERT(int(stage) < NO_STAGES);
            measurements[int(stage)] = time;
        }

        TIME_MEASURE_T get(STAGE_ENUM stage) const {
            return measurements[int(stage)];
        }

        TIME_MEASURE_T get_sum() const {
            TIME_MEASURE_T sum = 0.0;
            for (auto m : measurements) {
                sum += m;
            }
            return sum;
        }

        void clear() {
            for (auto& m : measurements)
                m = 0.0;
        }

        const measurements_t& get_all() const { return measurements; }

    private:
        measurements_t measurements;
};

template<typename STAGE_ENUM, size_t NO_STAGES, typename TIME_MEASURE_T>
class PerformanceRecorder
{
    public:
        using stage_names_t = std::array<const char*, NO_STAGES>;
        using StagePerformanceRecorder = SimplePerformanceRecorder<STAGE_ENUM, NO_STAGES, TIME_MEASURE_T>;
        using stage_col_widths_t = std::array<int, NO_STAGES>;

        static const int MIN_COL_WIDTH = 12, COL_PADDING = 3;

        PerformanceRecorder(const stage_names_t& stage_names, size_t iterations_prevues)
            : stage_names(stage_names), current_iteration(0) {

            stage_recorders.resize(std::max(iterations_prevues, size_t(1)));

            for (int i = 0; i < NO_STAGES; ++i) {
                stage_col_widths[i] = std::max(MIN_COL_WIDTH, COL_PADDING+int(strlen(stage_names[i])));
            }
        }

        void next_iteration() {
            ++current_iteration;
            if (current_iteration >= stage_recorders.size()) {
                stage_recorders.emplace_back(StagePerformanceRecorder());
            }
        }

        void register_measurement(STAGE_ENUM stage, TIME_MEASURE_T time) {
            ASSERT(int(stage) < NO_STAGES);
            ASSERT(current_iteration < stage_recorders.size());
            stage_recorders[current_iteration].set(stage, time);
        }

        StagePerformanceRecorder get_aggregated_sums() const {
            StagePerformanceRecorder aggregation;

            for (const auto& recorder : stage_recorders) {
                for (int stage = 0; stage < NO_STAGES; ++stage) {
                    aggregation.set(STAGE_ENUM(stage), aggregation.get_all()[stage] + recorder.get_all()[stage]);
                }
            }
            return aggregation;
        }


        void print_labels() const {
            for (int stage = 0; stage < NO_STAGES; ++stage) {
                std::cout << std::left << std::setw(stage_col_widths[stage]) << stage_names[stage];
            }
            std::cout << std::left << std::setw(MIN_COL_WIDTH) << "sum" << std::endl;
        }

        void print_recorder(const StagePerformanceRecorder& rec) const {
            std::cout << std::fixed << std::setprecision(3);
            for (int stage = 0; stage < NO_STAGES; ++stage) {
                std::cout << std::left << std::setw(stage_col_widths[stage]) << rec.get_all()[stage];
            }
            double sum = rec.get_sum();
            std::cout << std::left << std::setw(MIN_COL_WIDTH) << sum << std::endl;
        }

        void done() {
            aggregated = get_aggregated_sums();
        }

        TIME_MEASURE_T get_total() const {
            if (current_iteration > 0)  {
                return aggregated.get_sum();
            }
            else if(!stage_recorders.empty()) {
                return stage_recorders[0].get_sum();
            }
            return 0.0;
        }

        void print_single_rec_wrapped() const {
            const StagePerformanceRecorder& rec = stage_recorders[0];
            int max_width = 0;

            for (int stage = 0; stage < NO_STAGES; ++stage) {
                max_width = std::max(max_width, stage_col_widths[stage]);
            }

            for (int stage = 0; stage < NO_STAGES; ++stage) {
                std::cout << std::left << std::setw(max_width) << stage_names[stage] << ": ";
                std::cout << std::left << std::setw(max_width) << rec.get_all()[stage] << std::endl;
            }
            double sum = rec.get_sum();
            std::cout << std::left << std::setw(max_width) << "SUM " << ": " << sum << std::endl;
        }

        void print() const {
            if (current_iteration == 0) {
                print_single_rec_wrapped();
                return;
            }
            std::cout << std::endl;

            print_labels();

            if (stage_recorders.empty())
                return;

            for (int rep = 0; rep <= current_iteration; ++rep) {
                const StagePerformanceRecorder& r = stage_recorders[rep];
                print_recorder(r);
            }

            if (current_iteration > 0)  {
                std::cout << std::endl << "Aggregated:" << std::endl;
                print_labels();
                print_recorder(aggregated);
            }
        }

    private:
        const stage_names_t& stage_names;
        stage_col_widths_t stage_col_widths;

        std::vector<StagePerformanceRecorder> stage_recorders;
        StagePerformanceRecorder aggregated;
        size_t current_iteration;
};

}

#endif // PERFORMANCERECORDER_H

