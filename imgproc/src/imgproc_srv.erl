-module(imgproc_srv).

-behaviour(gen_server).

-define(IMAGE_HEIGHT, 512).
-define(IMAGE_WIDTH, 512).

%%------------------------------------------------------------------------------
-export([start_link/0]).
-export([init/1, terminate/2, code_change/3]).
-export([handle_call/3, handle_cast/2, handle_info/2]).
%%------------------------------------------------------------------------------

-record(state, { }).

%%------------------------------------------------------------------------------
start_link() ->
  {ok, KernelSrc} = application:get_env(kernel_src),
  imgproc_nif:initialize(KernelSrc),
  gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init(_) ->
  {ok, #state{}}.

terminate(_, _) ->
  imgproc_nif:teardown(),
  ok.

receive_from(Sock) ->
  gen_server:call(?MODULE, {receive_from, Sock}).

error_from(Sock) ->
  gen_server:call(?MODULE, {error_from, Sock}).
%%------------------------------------------------------------------------------

handle_call({receive_from, Sock, Lim}, _From, S) ->
  spawn(imgproc_srv, input_channel, [Sock, Lim]),
  {reply, ok, S};
handle_call({error_from, _Sock}, _From, S) ->
  {reply, ok, S}.

handle_cast(_, _) ->
  not_implemented.

handle_info(_, _) ->
  not_implemented.

code_change(_, _, _) ->
  not_implemented.

%%------------------------------------------------------------------------------
%% Listen for incoming connections
%%------------------------------------------------------------------------------
srv_loop() ->
  {ok, Host} = application:get_env(host),
  {ok, Port} = application:get_env(port),
  Opt = [binary, {packet, 0}, {active, false}, {exit_on_close, false}],
  imgproc_info:log(?MODULE, "Server listening on port ~p", [Port]),
  case gen_tcp:listen(Port, Opt) of
    {ok, LSock} ->
      imgproc_info:log(?MODULE, "Initiating server loop", []),
      tcp_accept_loop(LSock);
    {error, _} = Error ->
      imgproc_info:log(?MODULE, "Error ~p", [Error])
  end.

tcp_accept_loop(LSock) ->
  {ok, Sock} = gen_tcp:accept(LSock),
  case gen_tcp:recv(Sock, 5) of
    {ok, <<Ack:8,PacketDim:32>>} ->
      case Ack of
	0 ->
	  imgproc_info:log(?MODULE, "Success - adding camera", []),
	  gen_tcp:send(Sock, list_to_binary([0])),
	  imgproc_srv:receive_from(Sock);
	1 ->
	  imgproc_info:log(?MODULE, "Error, ack incorrect", []),
	  gen_tcp:send(Sock, list_to_binary([1])),
	  imgproc_srv:error_from(Sock)
      end;
    {error, closed} ->
      imgproc_info:log(?MODULE, "Client closed connection prematurely", [])
  end.
	  
input_channel(Sock, Lim) ->
  imgproc_info:log(?MODULE, "Receiving data ...", []),
  case gen_tcp:recv(Sock, Lim) of
    {ok, Data} ->
      %% Process data here
      imgproc_nif:transform(Data, ?IMAGE_WIDTH, ?IMAGE_HEIGHT),
      input_channel(Sock, Lim);
    {error, closed} ->
      imgproc_info:log(?MODULE, "Closed connection", []),
      ok
  end.
