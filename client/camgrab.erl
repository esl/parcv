-module(camgrab).
-export([capture/0]).
-export([bin_to_hex/1, byte_to_hex/1]).
-on_load(load_nifs/0).

load_nifs() ->
	erlang:load_nif("./camgrab_nif", 0).

capture() ->
	{error, "NIF library not loaded"}.

bin_to_hex(Bin) when is_binary(Bin) ->
	"<< " ++ string:join([byte_to_hex(B) || << B >> <= Bin ],", ") ++ " >>".

byte_to_hex(<< N1:4, N2:4 >>) ->
	[erlang:integer_to_list(N1, 16), erlang:integer_to_list(N2, 16)].
