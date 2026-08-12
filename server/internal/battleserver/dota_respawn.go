package battleserver

// dotaHeroRespawnTable contains the current default Dota hero respawn times
// for displayed levels 1..20. The curve is intentionally a table: Dota makes
// jumps after ultimate levels instead of using a single linear formula.
var dotaHeroRespawnTable = [...]float64{
	12, 15, 18, 21, 24,
	26, 28, 30, 32, 34, 36,
	44, 46, 48, 50, 52, 54,
	65, 70, 75,
}

// dotaHeroRespawnDelay returns the default Dota hero respawn time in seconds.
// The server stores level zero-based, so level 0 maps to displayed level 1.
// The game currently caps avatars at level 20; clamping also keeps the timer
// safe if an imported save or test provides a higher internal level.
func dotaHeroRespawnDelay(hs *huntState) float64 {
	displayedLevel := int32(1)
	if hs != nil {
		displayedLevel = hs.level + 1
	}
	if displayedLevel < 1 {
		displayedLevel = 1
	}
	if displayedLevel > int32(len(dotaHeroRespawnTable)) {
		displayedLevel = int32(len(dotaHeroRespawnTable))
	}
	return dotaHeroRespawnTable[displayedLevel-1]
}
