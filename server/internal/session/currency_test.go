package session

import "testing"

func TestCurrencyDenominations(t *testing.T) {
	if BronzePerSilver != 100 || SilverPerGold != 100 || BronzePerGold != 10000 {
		t.Fatalf("currency denominations = bronze/silver=%d, silver/gold=%d, bronze/gold=%d; want 100/100/10000", BronzePerSilver, SilverPerGold, BronzePerGold)
	}
}
