-module(ocl).

-export([get_platform_ids/0, acquire_device/1, acquire_context/2]).
-export([create_float_array/1, print_float_array/2]).
-export([acquire_command_queue/2, create_float_buffer/2, create_program/3]).
-export([create_kernel/3, create_event/0, set_kernel_arg/4]).
-export([enqueue_nd_range_kernel/5, enqueue_read_buffer/6]).
-export([wait_for_events/1]).
-export([test/0]).

-on_load(init/0).

init() ->
  PrivDir = case code:priv_dir(?MODULE) of
              {error, bad_name} ->
	        EbinDir = filename:dirname(code:which(?MODULE)),
		AppPath = filename:dirname(EbinDir),
		filename:join(AppPath, "priv");
              Path ->
	        Path
              end,
  erlang:load_nif("." ++ filename:join(PrivDir, "ocl"), 0).

get_platform_ids() ->
  exit(nif_library_not_loaded).

acquire_device(_Platform) ->
  exit(nif_library_not_loaded).

acquire_context(_Platform, _Device) ->
  exit(nif_library_not_loaded).

acquire_command_queue(_Context, _Device) ->
  exit(nif_library_not_loaded).

create_float_array(_N) ->
  exit(nif_library_not_loaded).

print_float_array(_Arr, _Lim) ->
  exit(nif_library_not_loaded).

create_float_buffer(_Context, _FloatList) ->
  exit(nif_library_not_loaded).

create_program(_Context, _Device, _Source) ->
  exit(nif_library_not_loaded).

create_kernel(_Device, _Program, _KernelName) ->
  exit(nif_library_not_loaded).

set_kernel_arg(_Kernel, _I, _Type, _Value) ->
  exit(nif_library_not_loaded).

create_event() ->
  exit(nif_library_not_loaded).

enqueue_nd_range_kernel(_CommandQueue, _Kernel, _WorkDims, _Global, _Local) ->
  exit(nif_library_not_loaded).

enqueue_read_buffer(_CommandQueue, _Mem, _Bool, _Offset, _SizeInBytes, _FloatArr) ->
  exit(nif_library_not_loaded).

wait_for_events(_Events) ->
  exit(nif_library_not_loaded).

test_src() ->
  "__kernel void matvecmult_kern(\n" ++
  "   uint n,__global float* aa,__global float* b,__global float* c )\n" ++
  "{\n" ++
  "   int i = get_global_id(0);\n" ++
  "   int j;\n" ++ 
  "   float tmp = 0.0f;\n" ++ 
  "   for(j=0;j<n;j++) tmp += aa[i*n+j] * b[j];\n" ++
  "   c[i] = aa[i*n+i];\n" ++ 
  "}\n".

gen_n(0, I, A) -> lists:reverse(A);
gen_n(N, I, A) -> gen_n(N-1, I, [I|A]).

test_buff_1() ->
  gen_n(4*4,0.5,[]).

test_buff_2() ->
  gen_n(4,2.0,[]).

test_buff_3() ->
  gen_n(4,0.0,[]).

test() ->
  {ok, P} = get_platform_ids(),
  {ok, D} = acquire_device(P),
  {ok, C} = acquire_context(P, D),
  {ok, CQ} = acquire_command_queue(C, D),
  {ok, Fb0} = create_float_buffer(C, test_buff_1()),
  {ok, Fb1} = create_float_buffer(C, test_buff_2()),
  {ok, Fb2} = create_float_buffer(C, test_buff_3()),
  {ok, Fa0} = create_float_array(4),
  {ok, Prg} = create_program(C, D, test_src()),
  {ok, K} = create_kernel(D, Prg, "matvecmult_kern"),
  ok = set_kernel_arg(K, 0, cl_uint, 4),
  ok = set_kernel_arg(K, 1, cl_mem, Fb0),
  ok = set_kernel_arg(K, 2, cl_mem, Fb1),
  ok = set_kernel_arg(K, 3, cl_mem, Fb2),
  {ok, E1} = enqueue_nd_range_kernel(CQ, K, 1, [16], [4]),
  {ok, E2} = enqueue_read_buffer(CQ, Fb2, true, 0, 4*4, Fa0),
  ok = wait_for_events([E1, E2]),
  ok = print_float_array(Fa0, 4).


  