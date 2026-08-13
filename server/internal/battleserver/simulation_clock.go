package battleserver

import (
	"container/heap"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// simulationTimer is the common cancellation surface used by live wall-clock
// timers and the deterministic queue owned by a headless Assault environment.
type simulationTimer interface {
	Stop() bool
}

type battleClock interface {
	Now() float64
	After(time.Duration, func()) simulationTimer
}

// liveBattleClock preserves the timing behaviour of ordinary client matches.
type liveBattleClock struct {
	start time.Time
}

func (c *liveBattleClock) Now() float64 {
	return time.Since(c.start).Seconds() * dotaSimulationSpeed()
}

func (c *liveBattleClock) After(d time.Duration, fn func()) simulationTimer {
	return time.AfterFunc(dotaSimulationWallDuration(d), fn)
}

// manualBattleClock is advanced explicitly by AssaultEnv.Step. Events with an
// equal deadline execute in insertion order, making identical seeds/actions
// reproducible and avoiding one goroutine/timer per attack during rollouts.
type manualBattleClock struct {
	mu     sync.Mutex
	now    float64
	serial uint64
	events simulationEventHeap
}

type simulationEvent struct {
	at        float64
	serial    uint64
	fn        func()
	cancelled bool
	index     int
}

type manualSimulationTimer struct {
	clock *manualBattleClock
	event *simulationEvent
}

func (t *manualSimulationTimer) Stop() bool {
	if t == nil || t.clock == nil || t.event == nil {
		return false
	}
	t.clock.mu.Lock()
	defer t.clock.mu.Unlock()
	if t.event.cancelled || t.event.index < 0 {
		return false
	}
	t.event.cancelled = true
	return true
}

type simulationEventHeap []*simulationEvent

func (h simulationEventHeap) Len() int { return len(h) }
func (h simulationEventHeap) Less(i, j int) bool {
	if h[i].at == h[j].at {
		return h[i].serial < h[j].serial
	}
	return h[i].at < h[j].at
}
func (h simulationEventHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index, h[j].index = i, j
}
func (h *simulationEventHeap) Push(x any) {
	e := x.(*simulationEvent)
	e.index = len(*h)
	*h = append(*h, e)
}
func (h *simulationEventHeap) Pop() any {
	old := *h
	n := len(old)
	e := old[n-1]
	e.index = -1
	*h = old[:n-1]
	return e
}

func newManualBattleClock() *manualBattleClock {
	c := &manualBattleClock{}
	heap.Init(&c.events)
	return c
}

func (c *manualBattleClock) Now() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *manualBattleClock) After(d time.Duration, fn func()) simulationTimer {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.serial++
	e := &simulationEvent{
		at:     c.now + d.Seconds(),
		serial: c.serial,
		fn:     fn,
		index:  -1,
	}
	heap.Push(&c.events, e)
	return &manualSimulationTimer{clock: c, event: e}
}

// Advance moves virtual time and executes every due callback. Callbacks run
// without the clock mutex because battle callbacks acquire the instance mutex
// and may schedule more events themselves.
func (c *manualBattleClock) Advance(d time.Duration) {
	c.mu.Lock()
	target := c.now + d.Seconds()
	for c.events.Len() > 0 {
		e := c.events[0]
		if e.at > target {
			break
		}
		heap.Pop(&c.events)
		c.now = e.at
		cancelled, fn := e.cancelled, e.fn
		c.mu.Unlock()
		if !cancelled && fn != nil {
			fn()
		}
		c.mu.Lock()
	}
	c.now = target
	c.mu.Unlock()
}

// dotaSimulationSpeed is a development-only wall-clock multiplier. Manual
// headless clocks never consult it.
func dotaSimulationSpeed() float64 {
	raw := strings.TrimSpace(os.Getenv("TANAT_DOTA_SIM_SPEED"))
	if raw == "" {
		return 1
	}
	speed, err := strconv.ParseFloat(raw, 64)
	if err != nil || speed <= 0 {
		return 1
	}
	return speed
}

func dotaSimulationWallDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return d
	}
	wall := time.Duration(float64(d) / dotaSimulationSpeed())
	if wall < time.Millisecond {
		return time.Millisecond
	}
	return wall
}

func dotaSimulationTickerInterval() time.Duration {
	return dotaSimulationWallDuration(tickInterval)
}
