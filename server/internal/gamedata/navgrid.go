package gamedata

import "math"

// NavGrid is a rasterised walkability map: a bit per Cell-sized square of the
// scene ground plane, true where the avatar may stand. It is built offline from
// the scene's real walkable surface meshes (see navgrid_map40.go) and reduced to
// the connected region around the spawn, so it is authoritative even when a
// scene's PassibilityData polygon is authored in a mismatched frame.
type NavGrid struct {
	MinX, MinY, Cell float64
	W, H             int
	bits             []uint64 // bit index = i*H + j, LSB-first
	spawn            Vec2
}

func (g *NavGrid) cellWalkable(i, j int) bool {
	if i < 0 || i >= g.W || j < 0 || j >= g.H {
		return false
	}
	b := i*g.H + j
	return g.bits[b>>6]&(uint64(1)<<(uint(b)&63)) != 0
}

// Walkable reports whether world (x,y) falls on a walkable cell.
func (g *NavGrid) Walkable(x, y float64) bool {
	i := int(math.Floor((x - g.MinX) / g.Cell))
	j := int(math.Floor((y - g.MinY) / g.Cell))
	return g.cellWalkable(i, j)
}

// Spawn returns the scene's spawn point (Reborn_point) in world coordinates.
func (g *NavGrid) Spawn() (float64, float64) { return g.spawn.X, g.spawn.Y }

// Clip returns the farthest point along the segment from (fx,fy) toward (tx,ty)
// that is still walkable — the avatar walks straight until it meets a wall/void
// and stops there. If the start is off-grid the avatar stays put.
func (g *NavGrid) Clip(fx, fy, tx, ty float64) (float64, float64) {
	if !g.Walkable(fx, fy) {
		return fx, fy
	}
	dx, dy := tx-fx, ty-fy
	dist := math.Hypot(dx, dy)
	if dist == 0 {
		return fx, fy
	}
	const step = 0.5
	steps := int(dist / step)
	lastX, lastY := fx, fy
	for i := 1; i <= steps; i++ {
		t := float64(i) * step / dist
		x, y := fx+dx*t, fy+dy*t
		if !g.Walkable(x, y) {
			return lastX, lastY
		}
		lastX, lastY = x, y
	}
	if g.Walkable(tx, ty) {
		return tx, ty
	}
	return lastX, lastY
}
