-module(ocv_cli_server).

-behaviour(gen_server).

%% API
-export([start_link/0]).
-export([connect/3, disconnect/0, send_data/1]).

%% gen_server callbacks
-export([init/1, terminate/2, code_change/3]).
-export([handle_call/3, handle_cast/2, handle_info/2]).

%% gen_server state
-record(state, {id, socket, packet_dim, status}).

%%------------------------------------------------------------------------------
start_link() ->
  gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

connect(Host, Port, PacketDim) ->
  gen_server:call(?MODULE, {connect, Host, Port, PacketDim}).

disconnect() ->
  gen_server:call(?MODULE, disconnect).

send_data(Data) ->
  gen_server:call(?MODULE, {send_data, Data}).

%%------------------------------------------------------------------------------
init([]) ->
  {ok, []}.

%% callbacks
handle_call({connect, Host, Port, PacketDim}, _From, S0) ->
  case gen_tcp:connect(Host, Port, [binary, {packet, 0}]) of
    {ok, Sock} ->
      S1 = input_client_handshake(Sock, PacketDim, S0),
      {reply, S1#state.status, S1};
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
	  Msg = list_to_binary(S0#state.id ++ Data),
	  ok = gen_tcp:send(S0#state.socket, Msg),
	  {reply, ok, S0};
	false ->
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
  Header = list_to_binary([1] ++ cli_f:to_bytes(PacketDim)),
  ok = gen_tcp:send(Sock, Header),
  case gen_tcp:recv(Sock, 1) of
    {ok, <<1,Id:32>>} ->
      S0#state{id=Id, socket=Sock, packet_dim=PacketDim, status=connected};
    {ok, <<0>>} ->
      ok = gen_tcp:close(Sock),
      S0#state{packet_dim=PacketDim, status=disconnected};
    {error, closed} ->
      S0#state{packet_dim=PacketDim, status=disconnected}
  end.
