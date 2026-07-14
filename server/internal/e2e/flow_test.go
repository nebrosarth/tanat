// Package e2e exercises the full "minimal end-to-end path" the real client
// walks: Ctrl login -> hero create -> common|area_conf -> Battle CONNECT
// handshake, across BOTH servers wired together exactly like cmd/ctrlserver,
// over real HTTP and TCP sockets. It is the integration seam the per-package
// tests can't cover: that the port advertised in area_conf actually reaches a
// live Battle listener that completes the handshake.
package e2e

import (
	"bytes"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/battleserver"
	"tanatserver/internal/ctrlserver"
)

func TestLoginToBattleHandshake(t *testing.T) {
	// Battle server on an ephemeral port, sharing the Ctrl server's store.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	battlePort := int32(ln.Addr().(*net.TCPAddr).Port)

	cs := ctrlserver.New()
	cs.BattleHost = "127.0.0.1"
	cs.BattlePorts = []int32{battlePort}
	battle := battleserver.New(cs.Store)
	go battle.Serve(ln)

	ts := httptest.NewServer(cs.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	// 1. Login.
	login := post(t, url, amf.NewArray().
		Set("object", "user").Set("action", "login").
		Set("params", amf.NewArray().Set("email", "e2e@example.com").Set("passwd", "pw").Set("version", "3.4.1.27345")).
		Set("sess_uid", "0").Set("sess_key", "").Set("counter", int32(1)))
	loginResp, ok := login.GetArray("user|login")
	if !ok {
		t.Fatalf("no user|login in response")
	}
	sessKey, _ := loginResp.GetString("sess_key")
	userID, _ := loginResp.GetInt("id")

	// 2. Hero create (human, race=1).
	post(t, url, amf.NewArray().
		Set("object", "hero").Set("action", "create").
		Set("params", amf.NewArray().Set("race", int32(1)).Set("gender", int32(1)).
			Set("face", int32(0)).Set("hair", int32(0)).Set("dist_mark", int32(0)).
			Set("skin_color", int32(0)).Set("hair_color", int32(0))).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", int32(2)))

	// 3. area_conf -> learn where the Battle server is.
	area := post(t, url, amf.NewArray().
		Set("object", "common").Set("action", "area_conf").
		Set("params", amf.NewArray()).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", int32(3)))
	fields, ok := area.GetArray("common|area_conf")
	if !ok {
		t.Fatalf("no common|area_conf in response")
	}
	conf, _ := fields.GetArray("area_conf")
	if scene, _ := conf.GetString("scene"); scene != "cs_human" {
		t.Errorf("scene = %q, want cs_human", scene)
	}
	ip, _ := conf.GetString("ip")
	ports, _ := conf.GetArray("port")
	if len(ports.Dense) == 0 {
		t.Fatalf("area_conf advertised no ports")
	}
	advPort, _ := ports.Dense[0].(int32)
	if advPort != battlePort {
		t.Fatalf("area_conf advertised port %d, want live battle port %d", advPort, battlePort)
	}

	// 4. Connect to the advertised Battle server and complete CONNECT, exactly
	// as BattleServerConnection.SendConnect does.
	addr := net.JoinHostPort(ip, strconv.Itoa(int(advPort)))
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial advertised battle server %s: %v", addr, err)
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(3 * time.Second))

	if err := battleproto.Write(conn, battleproto.Packet{
		Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", userID).Set("pass", ""),
	}); err != nil {
		t.Fatalf("send CONNECT: %v", err)
	}
	reply, err := battleproto.NewReader(conn).Read()
	if err != nil {
		t.Fatalf("read CONNECT reply: %v", err)
	}
	if reply.Cmd != battleproto.CmdConnect || !reply.Status {
		t.Fatalf("CONNECT reply = {%s status=%v}, want {CONNECT status=true}", reply.Cmd.Name(), reply.Status)
	}
	if v, _ := reply.Args.GetInt("clientId"); v != userID {
		t.Errorf("battle selfPlayerId = %d, want user id %d", v, userID)
	}
	if _, ok := reply.Args.GetInt("battleId"); !ok {
		t.Errorf("CONNECT reply missing battleId")
	}
}

func post(t *testing.T, url string, root *amf.MixedArray) *amf.MixedArray {
	t.Helper()
	var buf bytes.Buffer
	if err := amf.NewEncoder().EncodeMessage(&buf, root); err != nil {
		t.Fatalf("encode: %v", err)
	}
	resp, err := http.Post(url, "application/octet-stream", &buf)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	v, err := amf.NewDecoder(resp.Body).DecodeMessage()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	m, ok := v.(*amf.MixedArray)
	if !ok {
		t.Fatalf("response root is %T", v)
	}
	return m
}

