// Package ctrlproto implements the envelope format used on the Ctrl (HTTP)
// channel, as reverse engineered from TanatKernel's CtrlEntryPoint.Pack,
// CtrlPacket and CtrlParser: every request/response body is a single
// top-level AMF array. Requests carry sess_uid/sess_key/counter plus an
// optional object/action/params triple; responses carry one or more
// "object|action" keyed sub-arrays.
package ctrlproto

import "tanatserver/internal/amf"

// CmdKey builds the "object|action" wire key (CtrlCmdId.ToCallPath uses '|').
func CmdKey(obj, action string) string {
	return obj + "|" + action
}

// StatusOK is the magic "success" status value CtrlPacketValidator checks for
// (`_packet.Status == 100`).
const StatusOK int32 = 100

// Request is a decoded client->server envelope.
type Request struct {
	Object  string
	Action  string
	Params  *amf.MixedArray
	SessUID string
	SessKey string
	Counter int32
	IsPing  bool // true when object/action are absent (keepalive ping)
}

func ParseRequest(root *amf.MixedArray) Request {
	req := Request{
		SessUID: root.StringOr("sess_uid", ""),
		SessKey: root.StringOr("sess_key", ""),
		Counter: root.IntOr("counter", 0),
	}
	obj, hasObj := root.GetString("object")
	action, hasAction := root.GetString("action")
	if !hasObj || !hasAction {
		req.IsPing = true
		return req
	}
	req.Object = obj
	req.Action = action
	if p, ok := root.GetArray("params"); ok {
		req.Params = p
	} else {
		req.Params = amf.NewArray()
	}
	return req
}

// Response accumulates one or more "object|action" -> fields entries to be
// sent back as a single top-level AMF array (CtrlParser.AddToResponses reads
// every key of the root array as a separate response packet).
type Response struct {
	root *amf.MixedArray
}

func NewResponse() *Response {
	return &Response{root: amf.NewArray()}
}

// Ack adds a bare {status:100} success entry for a fire-and-forget command
// (the client-side equivalent of HandlerMgr.RegisterValidation).
func (r *Response) Ack(obj, action string) *Response {
	return r.Add(obj, action, amf.NewArray().Set("status", StatusOK))
}

// Fail adds a {status, error} failure entry.
func (r *Response) Fail(obj, action string, errorCode int32) *Response {
	fields := amf.NewArray().Set("status", errorCode).Set("error", errorCode)
	return r.Add(obj, action, fields)
}

// Add adds a fully custom fields array for one object|action response,
// auto-setting status:100 if the caller didn't already set one.
func (r *Response) Add(obj, action string, fields *amf.MixedArray) *Response {
	if _, ok := fields.Assoc["status"]; !ok {
		fields.Set("status", StatusOK)
	}
	r.root.Set(CmdKey(obj, action), fields)
	return r
}

func (r *Response) Root() *amf.MixedArray {
	return r.root
}
