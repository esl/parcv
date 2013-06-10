-module(imgproc_info).

-export([log/3]).

%%------------------------------------------------------------------------------
log(Module, Str, P) ->
  error_logger:info_msg("[~p] " ++ Str, [Module] ++ P).
