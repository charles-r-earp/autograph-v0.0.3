
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa-gfx803
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx803"
	.protected	u8_to_f32       ; -- Begin function u8_to_f32
	.globl	u8_to_f32
	.p2align	8
	.type	u8_to_f32,@function
u8_to_f32:                              ; @u8_to_f32
u8_to_f32$local:
; %bb.0:
	s_load_dword s0, s[6:7], 0x10
	s_load_dword s1, s[4:5], 0x4
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v0, vcc, s8, v0
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[6:7], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s3
	v_add_u32_e32 v2, vcc, s2, v2
	v_addc_u32_e32 v3, vcc, v4, v3, vcc
	v_mov_b32_e32 v4, s1
	v_add_u32_e32 v0, vcc, s0, v0
	v_addc_u32_e32 v1, vcc, v4, v1, vcc
	flat_load_ubyte v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cvt_f32_ubyte0_e32 v0, v0
	v_mul_f32_e32 v0, 0x3b808081, v0
	flat_store_dword v[2:3], v0
BB0_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel u8_to_f32
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 9
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	u8_to_f32, .Lfunc_end0-u8_to_f32
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 132
; NumSgprs: 11
; NumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 5
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	u8_to_one_hot_f32 ; -- Begin function u8_to_one_hot_f32
	.globl	u8_to_one_hot_f32
	.p2align	8
	.type	u8_to_one_hot_f32,@function
u8_to_one_hot_f32:                      ; @u8_to_one_hot_f32
u8_to_one_hot_f32$local:
; %bb.0:
	s_load_dword s0, s[6:7], 0x18
	s_load_dword s1, s[4:5], 0x4
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v0, vcc, s8, v0
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_2
; %bb.1:
	s_load_dwordx2 s[0:1], s[6:7], 0x0
	s_load_dword s4, s[6:7], 0x8
	s_load_dwordx2 s[2:3], s[6:7], 0x10
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, s1
	v_mul_lo_u32 v3, v0, s4
	v_add_u32_e32 v0, vcc, s0, v0
	v_addc_u32_e32 v1, vcc, v2, v1, vcc
	flat_load_ubyte v0, v[0:1]
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v4, s3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e32 v1, vcc, v3, v0
	v_lshlrev_b64 v[0:1], 2, v[1:2]
	v_mov_b32_e32 v2, 1.0
	v_add_u32_e32 v0, vcc, s2, v0
	v_addc_u32_e32 v1, vcc, v4, v1, vcc
	flat_store_dword v[0:1], v2
BB1_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel u8_to_one_hot_f32
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 9
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	u8_to_one_hot_f32, .Lfunc_end1-u8_to_one_hot_f32
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 156
; NumSgprs: 11
; NumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 5
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	add             ; -- Begin function add
	.globl	add
	.p2align	8
	.type	add,@function
add:                                    ; @add
add$local:
; %bb.0:
	s_load_dword s0, s[6:7], 0x18
	s_load_dword s1, s[4:5], 0x4
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v0, vcc, s8, v0
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[6:7], 0x0
	s_load_dwordx4 s[4:7], s[6:7], 0x10
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, s3
	v_mov_b32_e32 v3, s5
	v_add_u32_e32 v2, vcc, s4, v0
	v_addc_u32_e32 v3, vcc, v3, v1, vcc
	v_add_u32_e32 v4, vcc, s2, v0
	v_addc_u32_e32 v5, vcc, v5, v1, vcc
	v_mov_b32_e32 v6, s1
	v_add_u32_e32 v0, vcc, s0, v0
	v_addc_u32_e32 v1, vcc, v6, v1, vcc
	flat_load_dword v0, v[0:1]
	flat_load_dword v1, v[4:5]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_f32_e32 v0, v0, v1
	flat_store_dword v[2:3], v0
BB2_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 9
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end2:
	.size	add, .Lfunc_end2-add
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 152
; NumSgprs: 11
; NumVgprs: 7
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 7
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	cross_entropy_forward ; -- Begin function cross_entropy_forward
	.globl	cross_entropy_forward
	.p2align	8
	.type	cross_entropy_forward,@function
cross_entropy_forward:                  ; @cross_entropy_forward
cross_entropy_forward$local:
; %bb.0:
	s_load_dwordx2 s[2:3], s[6:7], 0x0
	s_load_dword s0, s[4:5], 0x4
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s8, s8, s0
	v_add_u32_e32 v0, vcc, s8, v0
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_14
; %bb.1:
	v_mul_lo_u32 v0, v0, s3
	s_load_dwordx4 s[8:11], s[6:7], 0x8
	s_load_dwordx2 s[4:5], s[6:7], 0x18
	v_mov_b32_e32 v1, 0
	s_cmp_lt_u32 s3, 2
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s9
	v_add_u32_e32 v1, vcc, s8, v1
	v_addc_u32_e32 v2, vcc, v3, v2, vcc
	flat_load_dword v2, v[1:2]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v1, v2
	s_cbranch_scc1 BB3_4
; %bb.2:
	s_add_i32 s0, s3, -1
	v_add_u32_e32 v3, vcc, 1, v0
	v_mov_b32_e32 v1, v2
BB3_3:                                  ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v4, 0
	v_lshlrev_b64 v[4:5], 2, v[3:4]
	v_add_u32_e32 v3, vcc, 1, v3
	v_mov_b32_e32 v6, s9
	v_add_u32_e32 v4, vcc, s8, v4
	v_addc_u32_e32 v5, vcc, v6, v5, vcc
	flat_load_dword v4, v[4:5]
	s_add_i32 s0, s0, -1
	v_mul_f32_e32 v1, 1.0, v1
	s_cmp_eq_u32 s0, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mul_f32_e32 v4, 1.0, v4
	v_max_f32_e32 v1, v4, v1
	s_cbranch_scc0 BB3_3
BB3_4:
	v_cmp_ne_u32_e64 s[0:1], s3, 0
	s_cmp_eq_u32 s3, 0
	v_mov_b32_e32 v3, 0
	s_cbranch_scc1 BB3_11
; %bb.5:
	v_mov_b32_e32 v3, v0
	s_mov_b32 s2, s3
	s_branch BB3_7
BB3_6:                                  ;   in Loop: Header=BB3_7 Depth=1
	v_lshlrev_b64 v[4:5], 2, v[3:4]
	v_mov_b32_e32 v2, s9
	v_add_u32_e32 v4, vcc, s8, v4
	v_addc_u32_e32 v5, vcc, v2, v5, vcc
	flat_load_dword v2, v[4:5]
	s_mov_b64 s[6:7], 0
	s_andn2_b64 vcc, exec, s[6:7]
	s_cbranch_vccz BB3_9
BB3_7:                                  ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v4, 0
	v_lshlrev_b64 v[5:6], 2, v[3:4]
	s_add_i32 s2, s2, -1
	v_mov_b32_e32 v7, s5
	v_add_u32_e32 v5, vcc, s4, v5
	v_addc_u32_e32 v6, vcc, v7, v6, vcc
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_f32_e32 v2, v2, v1
	s_cmp_eq_u32 s2, 0
	v_add_u32_e32 v3, vcc, 1, v3
	flat_store_dword v[5:6], v2
	s_cbranch_scc0 BB3_6
; %bb.8:                                ;   in Loop: Header=BB3_7 Depth=1
	s_mov_b64 s[6:7], -1
                                        ; implicit-def: $vgpr2
	s_andn2_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz BB3_7
BB3_9:
	v_mov_b32_e32 v3, 0
	s_mov_b32 s2, 0x39a3b295
	s_mov_b32 s6, 0x3fb8a000
	s_mov_b32 s7, 0xc2aeac50
	s_mov_b32 s8, 0x42b17218
	v_mov_b32_e32 v1, v0
	s_mov_b32 s9, s3
BB3_10:                                 ; =>This Inner Loop Header: Depth=1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64 v[4:5], 2, v[1:2]
	v_mov_b32_e32 v2, s5
	v_add_u32_e32 v4, vcc, s4, v4
	v_addc_u32_e32 v5, vcc, v2, v5, vcc
	flat_load_dword v2, v[4:5]
	s_add_i32 s9, s9, -1
	s_cmp_eq_u32 s9, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_and_b32_e32 v4, 0xfffff000, v2
	v_sub_f32_e32 v5, v2, v4
	v_mul_f32_e32 v7, s2, v5
	v_mac_f32_e32 v7, s6, v5
	v_mul_f32_e32 v6, s6, v4
	v_mac_f32_e32 v7, s2, v4
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v4, v7
	v_cmp_ngt_f32_e32 vcc, s7, v2
	v_mov_b32_e32 v5, 0x7f800000
	v_mul_f32_e32 v4, v6, v4
	v_cndmask_b32_e32 v4, 0, v4, vcc
	v_cmp_nlt_f32_e32 vcc, s8, v2
	v_cndmask_b32_e32 v2, v5, v4, vcc
	v_add_f32_e32 v3, v3, v2
	v_add_u32_e32 v1, vcc, 1, v1
	s_cbranch_scc0 BB3_10
BB3_11:
	v_log_f32_e32 v1, v3
	s_mov_b32 s6, 0x3805fdf4
	s_mov_b32 s7, 0x3f317000
	s_movk_i32 s2, 0x207
	v_and_b32_e32 v3, 0xfffff000, v1
	v_sub_f32_e32 v4, v1, v3
	v_mul_f32_e32 v2, s6, v4
	v_mac_f32_e32 v2, s6, v3
	v_mac_f32_e32 v2, s7, v4
	s_andn2_b64 vcc, exec, s[0:1]
	v_mac_f32_e32 v2, s7, v3
	v_cmp_class_f32_e64 s[0:1], v1, s2
	s_cbranch_vccnz BB3_14
; %bb.12:
	v_cndmask_b32_e64 v2, v2, v1, s[0:1]
BB3_13:                                 ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[3:4], 2, v[0:1]
	v_mov_b32_e32 v1, s5
	v_add_u32_e32 v5, vcc, s4, v3
	v_addc_u32_e32 v6, vcc, v1, v4, vcc
	v_mov_b32_e32 v7, s11
	v_add_u32_e32 v3, vcc, s10, v3
	flat_load_dword v1, v[5:6]
	v_addc_u32_e32 v4, vcc, v7, v4, vcc
	flat_load_dword v3, v[3:4]
	s_add_i32 s3, s3, -1
	s_cmp_lg_u32 s3, 0
	v_add_u32_e32 v0, vcc, 1, v0
	s_waitcnt vmcnt(1) lgkmcnt(1)
	v_sub_f32_e32 v1, v2, v1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mul_f32_e32 v1, v1, v3
	flat_store_dword v[5:6], v1
	s_cbranch_scc1 BB3_13
BB3_14:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel cross_entropy_forward
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end3:
	.size	cross_entropy_forward, .Lfunc_end3-cross_entropy_forward
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 680
; NumSgprs: 14
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 8
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	cross_entropy_backward ; -- Begin function cross_entropy_backward
	.globl	cross_entropy_backward
	.p2align	8
	.type	cross_entropy_backward,@function
cross_entropy_backward:                 ; @cross_entropy_backward
cross_entropy_backward$local:
; %bb.0:
	s_load_dword s0, s[6:7], 0x20
	s_load_dword s1, s[4:5], 0x4
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v0, vcc, s8, v0
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB4_2
; %bb.1:
	s_load_dwordx8 s[0:7], s[6:7], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s1
	v_add_u32_e32 v2, vcc, s0, v0
	v_addc_u32_e32 v3, vcc, v3, v1, vcc
	flat_load_dword v4, v[2:3]
	v_mov_b32_e32 v3, s5
	v_add_u32_e32 v2, vcc, s4, v0
	v_addc_u32_e32 v3, vcc, v3, v1, vcc
	flat_load_dword v2, v[2:3]
	s_load_dword s6, s[6:7], 0x0
	v_mov_b32_e32 v5, s3
	v_add_u32_e32 v0, vcc, s2, v0
	v_addc_u32_e32 v1, vcc, v5, v1, vcc
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_f32_e32 v2, v4, v2
	v_mul_f32_e32 v2, s6, v2
	flat_store_dword v[0:1], v2
BB4_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel cross_entropy_backward
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 9
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end4:
	.size	cross_entropy_backward, .Lfunc_end4-cross_entropy_backward
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 156
; NumSgprs: 11
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 6
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	reduce_sum_partial ; -- Begin function reduce_sum_partial
	.globl	reduce_sum_partial
	.p2align	8
	.type	reduce_sum_partial,@function
reduce_sum_partial:                     ; @reduce_sum_partial
reduce_sum_partial$local:
; %bb.0:
	s_load_dwordx4 s[0:3], s[6:7], 0x0
	s_load_dword s6, s[6:7], 0x10
	s_load_dword s4, s[4:5], 0x4
	s_mov_b32 m0, -1
	s_waitcnt lgkmcnt(0)
	s_and_b32 s4, s4, 0xffff
	s_mul_i32 s5, s8, s4
	s_lshl_b32 s7, s5, 1
	v_add_u32_e32 v1, vcc, s7, v0
	v_cmp_le_u32_e32 vcc, s6, v1
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[10:11], exec, s[10:11]
; %bb.1:
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v3, 0
	ds_write_b32 v2, v3
; %bb.2:
	s_or_saveexec_b64 s[10:11], s[10:11]
	s_xor_b64 exec, exec, s[10:11]
	s_cbranch_execz BB5_4
; %bb.3:
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	v_mov_b32_e32 v3, s1
	v_add_u32_e32 v1, vcc, s0, v1
	v_addc_u32_e32 v2, vcc, v3, v2, vcc
	flat_load_dword v1, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_write_b32 v2, v1
BB5_4:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v1, vcc, s4, v0
	v_add_u32_e32 v2, vcc, s7, v1
	v_cmp_le_u32_e32 vcc, s6, v2
	s_mov_b32 m0, -1
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[10:11], exec, s[10:11]
; %bb.5:
	v_lshlrev_b32_e32 v3, 2, v1
	v_mov_b32_e32 v4, 0
	ds_write_b32 v3, v4
; %bb.6:
	s_or_saveexec_b64 s[10:11], s[10:11]
	s_xor_b64 exec, exec, s[10:11]
	s_cbranch_execz BB5_8
; %bb.7:
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v4, s1
	v_add_u32_e32 v2, vcc, s0, v2
	v_addc_u32_e32 v3, vcc, v4, v3, vcc
	flat_load_dword v2, v[2:3]
	v_lshlrev_b32_e32 v1, 2, v1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_write_b32 v1, v2
BB5_8:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v1, vcc, s5, v0
	v_lshlrev_b32_e32 v2, 2, v0
	s_mov_b32 m0, -1
	s_branch BB5_11
BB5_9:                                  ;   in Loop: Header=BB5_11 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_lshr_b32 s4, s4, 1
	s_mov_b64 s[0:1], 0
BB5_10:                                 ;   in Loop: Header=BB5_11 Depth=1
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz BB5_14
BB5_11:                                 ; =>This Inner Loop Header: Depth=1
	s_cmp_lg_u32 s4, 0
	s_mov_b64 s[0:1], -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_cbranch_scc0 BB5_10
; %bb.12:                               ;   in Loop: Header=BB5_11 Depth=1
	v_cmp_gt_u32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB5_9
; %bb.13:                               ;   in Loop: Header=BB5_11 Depth=1
	s_lshl_b32 s5, s4, 2
	v_add_u32_e32 v3, vcc, s5, v2
	ds_read_b32 v3, v3
	ds_read_b32 v4, v2
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v3, v3, v4
	ds_write_b32 v2, v3
	s_branch BB5_9
BB5_14:
	v_cmp_eq_u32_e32 vcc, 0, v0
	v_lshlrev_b32_e32 v0, 1, v1
	v_cmp_gt_u32_e64 s[0:1], s6, v0
	s_mov_b32 s9, 0
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz BB5_16
; %bb.15:
	v_mov_b32_e32 v0, 0
	ds_read_b32 v2, v0
	s_lshl_b64 s[0:1], s[8:9], 2
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	s_waitcnt lgkmcnt(0)
	flat_store_dword v[0:1], v2
BB5_16:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel reduce_sum_partial
		.amdhsa_group_segment_fixed_size 2048
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end5:
	.size	reduce_sum_partial, .Lfunc_end5-reduce_sum_partial
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 444
; NumSgprs: 14
; NumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 2048 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 5
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	reduce_sum_final ; -- Begin function reduce_sum_final
	.globl	reduce_sum_final
	.p2align	8
	.type	reduce_sum_final,@function
reduce_sum_final:                       ; @reduce_sum_final
reduce_sum_final$local:
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_load_dword s4, s[4:5], 0x10
	v_mov_b32_e32 v0, 0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s2
	v_mov_b32_e32 v2, s3
	s_cmp_eq_u32 s4, 0
	flat_store_dword v[1:2], v0
	s_cbranch_scc1 BB6_2
BB6_1:                                  ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v1, s0
	flat_load_dword v3, v[1:2]
	s_add_i32 s4, s4, -1
	s_add_u32 s0, s0, 4
	s_addc_u32 s1, s1, 0
	v_mov_b32_e32 v1, s2
	v_mov_b32_e32 v2, s3
	s_cmp_eq_u32 s4, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_f32_e32 v0, v3, v0
	flat_store_dword v[1:2], v0
	s_cbranch_scc0 BB6_1
BB6_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel reduce_sum_final
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 6
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end6:
	.size	reduce_sum_final, .Lfunc_end6-reduce_sum_final
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 112
; NumSgprs: 6
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 6
; NumVGPRsForWavesPerEU: 4
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	reverse_conv_filter ; -- Begin function reverse_conv_filter
	.globl	reverse_conv_filter
	.p2align	8
	.type	reverse_conv_filter,@function
reverse_conv_filter:                    ; @reverse_conv_filter
reverse_conv_filter$local:
; %bb.0:
	s_load_dwordx2 s[0:1], s[6:7], 0x18
	s_load_dword s2, s[4:5], 0x4
	s_waitcnt lgkmcnt(0)
	s_and_b32 s2, s2, 0xffff
	s_mul_i32 s8, s8, s2
	v_add_u32_e32 v0, vcc, s8, v0
	v_cmp_gt_u32_e32 vcc, s1, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz BB7_10
; %bb.1:
	s_load_dwordx2 s[2:3], s[6:7], 0x0
	s_load_dword s1, s[6:7], 0x8
	s_load_dwordx2 s[4:5], s[6:7], 0x10
	v_cmp_ne_u32_e64 s[6:7], s0, 0
	s_mov_b64 s[8:9], -1
	s_waitcnt lgkmcnt(0)
	v_cmp_eq_f32_e64 s[10:11], s1, 0
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz BB7_6
; %bb.2:
	s_andn2_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz BB7_5
; %bb.3:
	v_add_u32_e32 v1, vcc, 1, v0
	v_mul_lo_u32 v2, s0, v1
	v_mul_lo_u32 v1, v0, s0
	s_mov_b32 s8, s0
	v_add_u32_e32 v3, vcc, -1, v2
BB7_4:                                  ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v4, 0
	v_lshlrev_b64 v[5:6], 2, v[3:4]
	v_mov_b32_e32 v2, s3
	v_add_u32_e32 v5, vcc, s2, v5
	v_addc_u32_e32 v6, vcc, v2, v6, vcc
	v_mov_b32_e32 v2, v4
	flat_load_dword v6, v[5:6]
	v_lshlrev_b64 v[4:5], 2, v[1:2]
	v_mov_b32_e32 v2, s5
	v_add_u32_e32 v4, vcc, s4, v4
	v_addc_u32_e32 v5, vcc, v2, v5, vcc
	flat_load_dword v2, v[4:5]
	s_add_i32 s8, s8, -1
	v_add_u32_e32 v3, vcc, -1, v3
	s_cmp_lg_u32 s8, 0
	v_add_u32_e32 v1, vcc, 1, v1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mac_f32_e32 v6, s1, v2
	flat_store_dword v[4:5], v6
	s_cbranch_scc1 BB7_4
BB7_5:
	s_mov_b64 s[8:9], 0
BB7_6:
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz BB7_10
; %bb.7:
	s_andn2_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz BB7_10
; %bb.8:
	v_add_u32_e32 v1, vcc, 1, v0
	v_mul_lo_u32 v1, s0, v1
	v_mul_lo_u32 v0, v0, s0
	v_add_u32_e32 v2, vcc, -1, v1
BB7_9:                                  ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[4:5], 2, v[2:3]
	v_mov_b32_e32 v1, s3
	v_add_u32_e32 v4, vcc, s2, v4
	v_addc_u32_e32 v5, vcc, v1, v5, vcc
	flat_load_dword v5, v[4:5]
	v_mov_b32_e32 v1, v3
	v_lshlrev_b64 v[3:4], 2, v[0:1]
	v_add_u32_e32 v2, vcc, -1, v2
	s_add_i32 s0, s0, -1
	v_mov_b32_e32 v6, s5
	v_add_u32_e32 v3, vcc, s4, v3
	v_addc_u32_e32 v4, vcc, v6, v4, vcc
	s_cmp_eq_u32 s0, 0
	v_add_u32_e32 v0, vcc, 1, v0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dword v[3:4], v5
	s_cbranch_scc0 BB7_9
BB7_10:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel reverse_conv_filter
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end7:
	.size	reverse_conv_filter, .Lfunc_end7-reverse_conv_filter
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 376
; NumSgprs: 14
; NumVgprs: 7
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 7
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	sgd_with_momentum ; -- Begin function sgd_with_momentum
	.globl	sgd_with_momentum
	.p2align	8
	.type	sgd_with_momentum,@function
sgd_with_momentum:                      ; @sgd_with_momentum
sgd_with_momentum$local:
; %bb.0:
	s_load_dword s0, s[6:7], 0x20
	s_load_dword s1, s[4:5], 0x4
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v0, vcc, s8, v0
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB8_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[6:7], 0x0
	s_load_dwordx2 s[4:5], s[6:7], 0x10
	s_load_dwordx2 s[6:7], s[6:7], 0x18
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, s3
	v_mov_b32_e32 v7, s1
	v_mov_b32_e32 v3, s7
	v_add_u32_e32 v2, vcc, s6, v0
	v_addc_u32_e32 v3, vcc, v3, v1, vcc
	v_add_u32_e32 v4, vcc, s2, v0
	v_addc_u32_e32 v5, vcc, v5, v1, vcc
	flat_load_dword v6, v[2:3]
	flat_load_dword v4, v[4:5]
	v_add_u32_e32 v0, vcc, s0, v0
	v_addc_u32_e32 v1, vcc, v7, v1, vcc
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mac_f32_e32 v4, s5, v6
	flat_store_dword v[2:3], v4
	flat_load_dword v2, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mad_f32 v2, -v4, s4, v2
	flat_store_dword v[0:1], v2
BB8_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel sgd_with_momentum
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 9
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end8:
	.size	sgd_with_momentum, .Lfunc_end8-sgd_with_momentum
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 188
; NumSgprs: 11
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 8
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.ident	"clang version 11.0.0 (/src/external/llvm-project/clang 6c08b900599eee52e12bce1e76b20dc413ce30e7)"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i8
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         16
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         24
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           u8_to_f32
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         u8_to_f32.kd
    .vgpr_count:     5
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     i8
      - .offset:         8
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           u8_to_one_hot_f32
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         u8_to_one_hot_f32.kd
    .vgpr_count:     5
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           add
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         add.kd
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         4
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           cross_entropy_forward
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         cross_entropy_forward.kd
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         88
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           cross_entropy_backward
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         cross_entropy_backward.kd
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         16
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         24
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 2048
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           reduce_sum_partial
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         reduce_sum_partial.kd
    .vgpr_count:     5
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         16
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         24
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           reduce_sum_final
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .sgpr_spill_count: 0
    .symbol:         reduce_sum_final.kd
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         8
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           reverse_conv_filter
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         reverse_conv_filter.kd
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         16
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .offset:         20
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .offset:         32
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         88
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           sgd_with_momentum
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         sgd_with_momentum.kd
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa-gfx803

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
	.text
	.file	"kernels.hip"
	.globl	__device_stub__u8_to_f32 # -- Begin function __device_stub__u8_to_f32
	.p2align	4, 0x90
	.type	__device_stub__u8_to_f32,@function
__device_stub__u8_to_f32:               # @__device_stub__u8_to_f32
.L__device_stub__u8_to_f32$local:
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$.L__device_stub__u8_to_f32$local, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end0:
	.size	__device_stub__u8_to_f32, .Lfunc_end0-__device_stub__u8_to_f32
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__u8_to_one_hot_f32 # -- Begin function __device_stub__u8_to_one_hot_f32
	.p2align	4, 0x90
	.type	__device_stub__u8_to_one_hot_f32,@function
__device_stub__u8_to_one_hot_f32:       # @__device_stub__u8_to_one_hot_f32
.L__device_stub__u8_to_one_hot_f32$local:
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movl	%esi, 12(%rsp)
	movq	%rdx, 64(%rsp)
	movl	%ecx, 8(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$.L__device_stub__u8_to_one_hot_f32$local, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end1:
	.size	__device_stub__u8_to_one_hot_f32, .Lfunc_end1-__device_stub__u8_to_one_hot_f32
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__add      # -- Begin function __device_stub__add
	.p2align	4, 0x90
	.type	__device_stub__add,@function
__device_stub__add:                     # @__device_stub__add
.L__device_stub__add$local:
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$.L__device_stub__add$local, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end2:
	.size	__device_stub__add, .Lfunc_end2-__device_stub__add
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__cross_entropy_forward # -- Begin function __device_stub__cross_entropy_forward
	.p2align	4, 0x90
	.type	__device_stub__cross_entropy_forward,@function
__device_stub__cross_entropy_forward:   # @__device_stub__cross_entropy_forward
.L__device_stub__cross_entropy_forward$local:
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movl	%edi, 4(%rsp)
	movl	%esi, (%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 80(%rsp)
	movq	%rsp, %rax
	movq	%rax, 88(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$.L__device_stub__cross_entropy_forward$local, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end3:
	.size	__device_stub__cross_entropy_forward, .Lfunc_end3-__device_stub__cross_entropy_forward
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__cross_entropy_backward # -- Begin function __device_stub__cross_entropy_backward
	.p2align	4, 0x90
	.type	__device_stub__cross_entropy_backward,@function
__device_stub__cross_entropy_backward:  # @__device_stub__cross_entropy_backward
.L__device_stub__cross_entropy_backward$local:
	.cfi_startproc
# %bb.0:
	subq	$136, %rsp
	.cfi_def_cfa_offset 144
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	96(%rsp), %r9
	movl	$.L__device_stub__cross_entropy_backward$local, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$152, %rsp
	.cfi_adjust_cfa_offset -152
	retq
.Lfunc_end4:
	.size	__device_stub__cross_entropy_backward, .Lfunc_end4-__device_stub__cross_entropy_backward
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__reduce_sum_partial # -- Begin function __device_stub__reduce_sum_partial
	.p2align	4, 0x90
	.type	__device_stub__reduce_sum_partial,@function
__device_stub__reduce_sum_partial:      # @__device_stub__reduce_sum_partial
.L__device_stub__reduce_sum_partial$local:
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$.L__device_stub__reduce_sum_partial$local, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end5:
	.size	__device_stub__reduce_sum_partial, .Lfunc_end5-__device_stub__reduce_sum_partial
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__reduce_sum_final # -- Begin function __device_stub__reduce_sum_final
	.p2align	4, 0x90
	.type	__device_stub__reduce_sum_final,@function
__device_stub__reduce_sum_final:        # @__device_stub__reduce_sum_final
.L__device_stub__reduce_sum_final$local:
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$.L__device_stub__reduce_sum_final$local, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end6:
	.size	__device_stub__reduce_sum_final, .Lfunc_end6-__device_stub__reduce_sum_final
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__reverse_conv_filter # -- Begin function __device_stub__reverse_conv_filter
	.p2align	4, 0x90
	.type	__device_stub__reverse_conv_filter,@function
__device_stub__reverse_conv_filter:     # @__device_stub__reverse_conv_filter
.L__device_stub__reverse_conv_filter$local:
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movss	%xmm0, 12(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 8(%rsp)
	movl	%ecx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$.L__device_stub__reverse_conv_filter$local, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end7:
	.size	__device_stub__reverse_conv_filter, .Lfunc_end7-__device_stub__reverse_conv_filter
	.cfi_endproc
                                        # -- End function
	.globl	__device_stub__sgd_with_momentum # -- Begin function __device_stub__sgd_with_momentum
	.p2align	4, 0x90
	.type	__device_stub__sgd_with_momentum,@function
__device_stub__sgd_with_momentum:       # @__device_stub__sgd_with_momentum
.L__device_stub__sgd_with_momentum$local:
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movss	%xmm0, 20(%rsp)
	movss	%xmm1, 16(%rsp)
	movq	%rdx, 72(%rsp)
	movl	%ecx, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	leaq	96(%rsp), %r9
	movl	$.L__device_stub__sgd_with_momentum$local, %edi
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end8:
	.size	__device_stub__sgd_with_momentum, .Lfunc_end8-__device_stub__sgd_with_momentum
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90         # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB9_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle(%rip)
.LBB9_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__u8_to_f32$local, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__u8_to_one_hot_f32$local, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__add$local, %esi
	movl	$.L__unnamed_3, %edx
	movl	$.L__unnamed_3, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__cross_entropy_forward$local, %esi
	movl	$.L__unnamed_4, %edx
	movl	$.L__unnamed_4, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__cross_entropy_backward$local, %esi
	movl	$.L__unnamed_5, %edx
	movl	$.L__unnamed_5, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__reduce_sum_partial$local, %esi
	movl	$.L__unnamed_6, %edx
	movl	$.L__unnamed_6, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__reduce_sum_final$local, %esi
	movl	$.L__unnamed_7, %edx
	movl	$.L__unnamed_7, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__reverse_conv_filter$local, %esi
	movl	$.L__unnamed_8, %edx
	movl	$.L__unnamed_8, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$.L__device_stub__sgd_with_momentum$local, %esi
	movl	$.L__unnamed_9, %edx
	movl	$.L__unnamed_9, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit                  # TAILCALL
.Lfunc_end9:
	.size	__hip_module_ctor, .Lfunc_end9-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90         # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	__hip_gpubin_handle(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB10_2
# %bb.1:
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle(%rip)
.LBB10_2:
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end10:
	.size	__hip_module_dtor, .Lfunc_end10-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	.L__unnamed_1,@object   # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"u8_to_f32"
	.size	.L__unnamed_1, 10

	.type	.L__unnamed_2,@object   # @1
.L__unnamed_2:
	.asciz	"u8_to_one_hot_f32"
	.size	.L__unnamed_2, 18

	.type	.L__unnamed_3,@object   # @2
.L__unnamed_3:
	.asciz	"add"
	.size	.L__unnamed_3, 4

	.type	.L__unnamed_4,@object   # @3
.L__unnamed_4:
	.asciz	"cross_entropy_forward"
	.size	.L__unnamed_4, 22

	.type	.L__unnamed_5,@object   # @4
.L__unnamed_5:
	.asciz	"cross_entropy_backward"
	.size	.L__unnamed_5, 23

	.type	.L__unnamed_6,@object   # @5
.L__unnamed_6:
	.asciz	"reduce_sum_partial"
	.size	.L__unnamed_6, 19

	.type	.L__unnamed_7,@object   # @6
.L__unnamed_7:
	.asciz	"reduce_sum_final"
	.size	.L__unnamed_7, 17

	.type	.L__unnamed_8,@object   # @7
.L__unnamed_8:
	.asciz	"reverse_conv_filter"
	.size	.L__unnamed_8, 20

	.type	.L__unnamed_9,@object   # @8
.L__unnamed_9:
	.asciz	"sgd_with_momentum"
	.size	.L__unnamed_9, 18

	.type	__hip_fatbin_wrapper,@object # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3
__hip_fatbin_wrapper:
	.long	1212764230              # 0x48495046
	.long	1                       # 0x1
	.quad	__hip_fatbin
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.hidden	__hip_gpubin_handle     # @__hip_gpubin_handle
	.type	__hip_gpubin_handle,@object
	.bss
	.weak	__hip_gpubin_handle
	.p2align	3
__hip_gpubin_handle:
	.quad	0
	.size	__hip_gpubin_handle, 8

	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	__hip_module_ctor
	.ident	"clang version 11.0.0 (/src/external/llvm-project/clang 6c08b900599eee52e12bce1e76b20dc413ce30e7)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __device_stub__u8_to_f32
	.addrsig_sym __device_stub__u8_to_one_hot_f32
	.addrsig_sym __device_stub__add
	.addrsig_sym __device_stub__cross_entropy_forward
	.addrsig_sym __device_stub__cross_entropy_backward
	.addrsig_sym __device_stub__reduce_sum_partial
	.addrsig_sym __device_stub__reduce_sum_final
	.addrsig_sym __device_stub__reverse_conv_filter
	.addrsig_sym __device_stub__sgd_with_momentum
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym __hip_fatbin
	.addrsig_sym __hip_fatbin_wrapper

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
