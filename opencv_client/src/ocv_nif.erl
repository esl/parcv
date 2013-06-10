-module(ocv_nif).

-export([any_device/0]).
-export([new_frame/1, query_frame/2]).
-export([frame_to_tuple/1]).
-export([loop_send_frames/0]).

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

send_frames(D, F) ->
  ok = query_frame(D, F),
  {W, H, NChan, ImageSize, Data} = frame_to_tuple(F),
  ocv_cli_server:send_data(Data),
  send_frames(D, F).

loop_send_frames() ->
  {ok, D} = any_device(),
  {ok, F} = new_frame(D),
  send_frames(D, F).

any_device() ->
  ?NIF_STUB.

new_frame(_Device) ->
  ?NIF_STUB.

query_frame(_Device, _Frame) ->
  ?NIF_STUB.

frame_to_tuple(_Frame) ->
  ?NIF_STUB.
