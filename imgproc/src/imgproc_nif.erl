-module(imgproc_nif).

-export([initialize/1, transform/3, teardown/0]).
-export([list_to_image/1, read_png/3, write_png/4]).

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
  erlang:load_nif("." ++ filename:join(PrivDir, "imgproc_drv"), 0).

initialize(_Kernel) ->
  ?NIF_STUB.

teardown() ->
  ?NIF_STUB.

transform(_Image, _ImageW, _ImageH) ->
  ?NIF_STUB.

list_to_image(_List) ->
  ?NIF_STUB.

read_png(_Filename, _ImageW, _ImageH) ->
  ?NIF_STUB.

write_png(_Filename, _Image, _ImageW, _ImageH) ->
  ?NIF_STUB.
		
  
