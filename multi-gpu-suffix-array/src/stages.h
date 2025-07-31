#ifndef STAGES_H
#define STAGES_H

namespace perf_rec {

#define MAIN_STAGES(decl_main_stage)       \   
    decl_main_stage(Produce_KMers)         \
        decl_main_stage(Initial_Sort)          \
        decl_main_stage(Initial_Merge)         \
        decl_main_stage(Initial_Ranking)       \
        decl_main_stage(Initial_Write_To_ISA)  \
        decl_main_stage(Initial_Compacting)    \
        decl_main_stage(Final_Transpose)       \
        decl_main_stage(Prepare_S12_for_Merge) \
        decl_main_stage(Prepare_S0_for_Merge)  \
        decl_main_stage(Final_Merge)

#define LOOP_STAGES(decl_loop_stage) \
    decl_loop_stage(Fetch_Rank)      \
    decl_loop_stage(Segmented_Sort)  \
    decl_loop_stage(Merge)           \
    decl_loop_stage(Rebucket)        \
    decl_loop_stage(Write_Isa)       \
    decl_loop_stage(Compacting)      \

#define WRITE_ISA_STAGES(decl_write_isa_stage) \
    decl_write_isa_stage(Multisplit)           \
    decl_write_isa_stage(All2All)              \
    decl_write_isa_stage(Sort)                 \
    decl_write_isa_stage(WriteIsa)

#define FETCH_RANK_STAGES(decl_fetch_rank_stage) \
    decl_fetch_rank_stage(Prepare_Indices)       \
    decl_fetch_rank_stage(Multisplit)            \
    decl_fetch_rank_stage(All2AllForth)          \
    decl_fetch_rank_stage(Fetch)                 \
    decl_fetch_rank_stage(All2AllBack)           \
    decl_fetch_rank_stage(WriteRanks)

#define PREPARE_FINAL_MERGE_STAGES(decl_prepare_final_merge_stage) \
    decl_prepare_final_merge_stage(S12_Multisplit)                 \
    decl_prepare_final_merge_stage(S12_Write_Out)                  \
    decl_prepare_final_merge_stage(S12_All2All)                    \
    decl_prepare_final_merge_stage(S12_Write_Into_Place)           \
    decl_prepare_final_merge_stage(S0_Write_Out_And_Sort)          \
    decl_prepare_final_merge_stage(S0_Merge)                       \
    decl_prepare_final_merge_stage(S0_Combine)



#define CREATE_STAGE_ENUM(name) name,

#define CREATE_STAGE_STRINGS(name) #name,

        enum class MainStages {
        MAIN_STAGES(CREATE_STAGE_ENUM)
        NO_STAGES
    };

    const std::array<const char*, size_t(MainStages::NO_STAGES)> main_stage_names = {
        MAIN_STAGES(CREATE_STAGE_STRINGS)
    };


    enum class LoopStages {
        LOOP_STAGES(CREATE_STAGE_ENUM)
        NO_STAGES
    };

    const std::array<const char*, size_t(LoopStages::NO_STAGES)> loop_stage_names = {
        LOOP_STAGES(CREATE_STAGE_STRINGS)
    };


    enum class WriteISAStages {
        WRITE_ISA_STAGES(CREATE_STAGE_ENUM)
        NO_STAGES
    };

    const std::array<const char*, size_t(WriteISAStages::NO_STAGES)> write_isa_stage_names = {
        WRITE_ISA_STAGES(CREATE_STAGE_STRINGS)
    };


    enum class FetchRankStages {
        FETCH_RANK_STAGES(CREATE_STAGE_ENUM)
        NO_STAGES
    };

    const std::array<const char*, size_t(FetchRankStages::NO_STAGES)> fetch_rank_stage_names = {
        FETCH_RANK_STAGES(CREATE_STAGE_STRINGS)
    };

    enum class PrepareFinalMergeStages {
        PREPARE_FINAL_MERGE_STAGES(CREATE_STAGE_ENUM)
        NO_STAGES
    };

    const std::array<const char*, size_t(PrepareFinalMergeStages::NO_STAGES)> prepare_final_merge_stages_names = {
        PREPARE_FINAL_MERGE_STAGES(CREATE_STAGE_STRINGS)
    };


}

#endif // STAGES_H
