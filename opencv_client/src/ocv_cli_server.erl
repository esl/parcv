-module(ocv_cli_server).

-behaviour(gen_server).

%% API
-export([start_link/0]).
-export([connect/3, disconnect/0, send_data/1]).

%% gen_server callbacks
-export([init/1, terminate/2, code_change/3]).
-export([handle_call/3, handle_cast/2, handle_info/2]).

%% gen_server state
-record(state, {id=undefined, socket=undefined, packet_dim=undefined, status=disconnected}).

%%------------------------------------------------------------------------------
start_link() ->
  {ok, SrvPid} = gen_server:start_link({local, ?MODULE}, ?MODULE, [], []),
  {ok, Host} = application:get_env(host),
  {ok, Port} = application:get_env(port),
  {ok, Size} = application:get_env(size),
  case connect(Host, Port, Size) of
    disconnected ->
      io:format("[ocl_cli_server] Disconnected~n"),
      disconnected;
    _Other ->
      _NifPid = spawn_link(ocv_nif, loop_send_frames, []),
      {ok, SrvPid}
  end.

connect(Host, Port, PacketDim) ->
  gen_server:call(?MODULE, {connect, Host, Port, PacketDim}).

disconnect() ->
  gen_server:call(?MODULE, disconnect).

send_data(Data) ->
  gen_server:call(?MODULE, {send_data, Data}).

%%------------------------------------------------------------------------------
init([]) ->
  {ok, #state{}}.

%% callbacks
handle_call({connect, Host, Port, PacketDim}, _From, S0) ->
  case gen_tcp:connect(Host, Port, [binary, {packet, 0}, {active, false}, {exit_on_close, false}]) of
    {ok, Sock} ->
      io:format("Initiating client handshake~n",[]),
      S1 = input_client_handshake(Sock, PacketDim, S0),
      io:format("Updating state to ~p~n", [S1]),
      {reply, ok, S1};
    {error, Reason} ->
      io:format("error: ~p", [Reason]),
      S1 = S0#state{
	     packet_dim = PacketDim,
	     status = disconnected
	    },
      {reply, disconnected, S1}
  end;

handle_call(disconnect, _From, S0) ->
  case S0#state.status of
    connected ->
      ok = gen_tcp:close(S0#state.socket),
      S1 = S0#state{status=disconnected},
      {reply, ok, S1};
    disconnected ->
      {reply, ok, S0}
  end;

handle_call({send_data, Data}, _From, S0) when is_list(Data) ->
  case S0#state.status of
    connected ->
      case length(Data) == S0#state.packet_dim of
	true ->
	  io:format("Sending data~n",[]),
	  Msg = list_to_binary(Data),
	  ok = gen_tcp:send(S0#state.socket, Msg),
	  case gen_tcp:recv(S0#state.socket, 0) of
	    {ok, <<0>>} ->
	      {reply, ok, S0};
	    Other ->
	      io:format("Error, received other than <<0>>, ~p~n", [Other]),
	      {reply, error, S0}
	  end;
	false ->
	  io:format("Data arity mismatch, data_sz: ~p, packet_dim: ~p~n", [length(Data), S0#state.packet_dim]),
	  {reply, data_arity_mismatch, S0}
      end;
    disconnected ->
      {reply, disconnected, S0}
  end.

handle_cast(_Msg, State) ->
  {noreply, State}.

handle_info(_Info, State) ->
  {noreply, State}.

terminate(_Reason, _State) ->
  ok.

code_change(_OldVsn, State, _Extra) ->
  {ok, State}.

%%------------------------------------------------------------------------------
%% Internal
%%------------------------------------------------------------------------------

%%------------------------------------------------------------------------------
%% @doc The handshake for a client which acts as an input channel to 
%% cls_acceptor. PacketDim refers to the size of the data packets which are to
%% be streamed to the server.
%%------------------------------------------------------------------------------
input_client_handshake(Sock, PacketDim, S0) ->
  Header = list_to_binary([0] ++ cli_f:to_bytes(PacketDim)),
  ok = gen_tcp:send(Sock, Header),
  case gen_tcp:recv(Sock, 1) of
    {ok, <<0>>} ->
      io:format("Received 0 ( success )~n",[]),
      S0#state{socket=Sock, packet_dim=PacketDim, status=connected};
    {ok, <<1>>} ->
      io:format("Received 1 ( failure )~n",[]),
      ok = gen_tcp:close(Sock),
      S0#state{packet_dim=PacketDim, status=disconnected};
    {error, closed} ->
      S0#state{packet_dim=PacketDim, status=disconnected};
    _ -> S0
  end.
