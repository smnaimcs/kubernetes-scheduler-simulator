package simontype

import (
	"math"
	"testing"
)

func TestGpuTypeToTier(t *testing.T) {
	tests := []struct {
		input    string
		wantTier GpuTier
		wantOk   bool
	}{
		{"A10", TierA10, true},
		{"G3", TierG3, true},
		{"V100M32", TierV100M32, true},
		{"V100M16", TierV100M16, true},
		{"G2", TierG2, true},
		{"T4", TierT4, true},
		{"P100", TierP100, true},
		{"", TierUnknown, false},
		{"UnknownGPU", TierUnknown, false},
		{"V100M16|V100M32", TierV100M32, true},
		{"A10|T4", TierA10, true},
	}

	for _, tt := range tests {
		gotTier, gotOk := GpuTypeToTier(tt.input)
		if gotTier != tt.wantTier || gotOk != tt.wantOk {
			t.Errorf("GpuTypeToTier(%q) = (%v, %v), want (%v, %v)",
				tt.input, gotTier, gotOk, tt.wantTier, tt.wantOk)
		}
	}
}

func TestTierMultiplier(t *testing.T) {
	const eps = 1e-6

	tests := []struct {
		name      string
		podTier   GpuTier
		nodeTier  GpuTier
		podKnown  bool
		nodeKnown bool
		wantMult  float64
	}{
		{
			name:      "perfect match",
			podTier:   TierV100M16,
			nodeTier:  TierV100M16,
			podKnown:  true,
			nodeKnown: true,
			wantMult:  1.00,
		},
		{
			name:      "unknown pod tier",
			podTier:   TierUnknown,
			nodeTier:  TierA10,
			podKnown:  false,
			nodeKnown: true,
			wantMult:  1.00,
		},
		{
			name:      "unknown node tier",
			podTier:   TierP100,
			nodeTier:  TierUnknown,
			podKnown:  true,
			nodeKnown: false,
			wantMult:  1.00,
		},
		{
			name:      "empty / unknown both",
			podTier:   TierUnknown,
			nodeTier:  TierUnknown,
			podKnown:  false,
			nodeKnown: false,
			wantMult:  1.00,
		},
		{
			name:      "max diff low-tier pod on stronger GPU (P100 on A10)",
			podTier:   TierP100, // 1
			nodeTier:  TierA10,  // 7, diff = 6, maxDiff = 6 -> 0.92 - 0.32*(6/6) = 0.60
			podKnown:  true,
			nodeKnown: true,
			wantMult:  0.60,
		},
		{
			name:      "max diff high-tier pod on weaker GPU (A10 on P100)",
			podTier:   TierA10,  // 7
			nodeTier:  TierP100, // 1, diff = 6, maxDiff = 6 -> 0.80 - 0.70*(6/6) = 0.10
			podKnown:  true,
			nodeKnown: true,
			wantMult:  0.10,
		},
		{
			name:      "partial diff low-tier pod on stronger GPU (T4 on G3)",
			podTier:   TierT4, // 2
			nodeTier:  TierG3, // 6, diff = 4 -> 0.92 - 0.32*(4/6) = 0.706666...
			podKnown:  true,
			nodeKnown: true,
			wantMult:  0.92 - 0.32*(4.0/6.0),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TierMultiplier(tt.podTier, tt.nodeTier, tt.podKnown, tt.nodeKnown)
			if math.Abs(got-tt.wantMult) > eps {
				t.Errorf("TierMultiplier(%v, %v, %v, %v) = %v, want %v",
					tt.podTier, tt.nodeTier, tt.podKnown, tt.nodeKnown, got, tt.wantMult)
			}
		})
	}
}
