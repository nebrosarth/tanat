package session

// Money is stored as one integer in the smallest client-visible denomination.
// The client decomposes it into gold, silver, and bronze when rendering the
// wallet: 100 bronze make one silver, and 100 silver make one gold.
const (
	BronzePerSilver int32 = 100
	SilverPerGold   int32 = 100
	BronzePerGold   int32 = BronzePerSilver * SilverPerGold
)
