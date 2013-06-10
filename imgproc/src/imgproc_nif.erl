-module(imgproc_nif).

-export([clinitialize/1, cltransform/3, clteardown/0]).
-export([cllist_to_image/1, clread_png/3, clwrite_png/4]).

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

clinitialize(_Kernel) ->
  ?NIF_STUB.

clteardown() ->
  ?NIF_STUB.

cltransform(_Image, _ImageW, _ImageH) ->
  ?NIF_STUB.

cllist_to_image(_List) ->
  ?NIF_STUB.

clread_png(_Filename, _ImageW, _ImageH) ->
  ?NIF_STUB.

clwrite_png(_Filename, _Image, _ImageW, _ImageH) ->
  ?NIF_STUB.
		
  
