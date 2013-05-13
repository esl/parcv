-module(ocv_nif).

-export([any_device/0, free_device/1]).
-export([new_frame/1, free_frame/1, query_frame/2]).
-export([frame_to_tuple/1]).

-on_load(init/0).

-define(NIF_STUB, exit(nif_library_not_loaded)).

init() ->
  PrivDir = case code:priv_dir(?MODULE) of
	      {error, bad_name} ->
		EbinDir = filename:dirname(code:which(?MODULE)),
		AppPath = filename:dirname(EbinDir),
		filename:join(AppPath, "priv");
	      Path ->
		Path
	    end,
  erlang:load_nif(filename:join(PrivDir, "ocv_drv"), 0).

any_device() ->
  ?NIF_STUB.

free_device(_Device) ->
  ?NIF_STUB.

new_frame(_Device) ->
  ?NIF_STUB.

free_frame(_Frame) ->
  ?NIF_STUB.

query_frame(_Device, _Frame) ->
  ?NIF_STUB.

frame_to_tuple(_Frame) ->
  ?NIF_STUB.
