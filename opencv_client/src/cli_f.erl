-module(cli_f).

-export([to_bytes/1]).

to_bytes(I) when is_integer(I) ->
  BI = binary:encode_unsigned(I),
  LI = binary_to_list(BI),
  to_bytes(BI, LI).

pad_list(LI, 0, Acc) ->
  Acc ++ LI;
pad_list(LI, N, Acc) ->
  pad_list(LI, N - 1, [0|Acc]).

to_bytes(_BI, LI) when is_list(LI) andalso length(LI) < 4 ->
  pad_list(LI, 4 - length(LI), []);
to_bytes(BI, LI) when is_list(LI) andalso length(LI) == 4 ->
  BI.

