package simontype

import (
	"strings"
)

type GpuTier int

const (
	TierUnknown GpuTier = 0
	TierP100    GpuTier = 1
	TierT4      GpuTier = 2
	TierG2      GpuTier = 3
	TierV100M16 GpuTier = 4
	TierV100M32 GpuTier = 5
	TierG3      GpuTier = 6
	TierA10     GpuTier = 7
)

const NumTiers = 7

// GpuTypeToTier maps a GPU model string (or pipe-separated list of model strings)
// to its corresponding GpuTier rank. Higher GpuTier value indicates a higher performance tier.
// Returns (0, false) if gpuType is empty or unknown.
func GpuTypeToTier(gpuType string) (GpuTier, bool) {
	if len(gpuType) == 0 {
		return TierUnknown, false
	}

	parts := strings.Split(gpuType, "|")
	var maxTier GpuTier = TierUnknown
	found := false

	for _, part := range parts {
		part = strings.TrimSpace(part)
		tier, ok := singleGpuTypeToTier(part)
		if ok {
			found = true
			if tier > maxTier {
				maxTier = tier
			}
		}
	}

	if !found {
		return TierUnknown, false
	}
	return maxTier, true
}

func singleGpuTypeToTier(model string) (GpuTier, bool) {
	switch model {
	case "A10":
		return TierA10, true
	case "G3":
		return TierG3, true
	case "V100M32":
		return TierV100M32, true
	case "V100M16":
		return TierV100M16, true
	case "G2":
		return TierG2, true
	case "T4":
		return TierT4, true
	case "P100":
		return TierP100, true
	default:
		return TierUnknown, false
	}
}

// TierMultiplier calculates the GPU tier matching soft multiplier m_tier.
// Returns 1.0 if either pod or node tier is unknown (!podKnown || !nodeKnown).
// Otherwise:
//   perfect match (diff == 0): m_tier = 1.00
//   low-tier pod on stronger GPU (nodeTier > podTier): m_tier = 0.92 - 0.32 * (diff / maxDiff)
//   high-tier pod on weaker GPU (podTier > nodeTier):  m_tier = 0.80 - 0.70 * (diff / maxDiff)
// where maxDiff = NumTiers - 1 = 6.
func TierMultiplier(podTier, nodeTier GpuTier, podKnown, nodeKnown bool) float64 {
	if !podKnown || !nodeKnown {
		return 1.0
	}

	if podTier == nodeTier {
		return 1.00
	}

	maxDiff := float64(NumTiers - 1) // 6.0

	if nodeTier > podTier {
		// Low-tier pod on stronger GPU
		diff := float64(nodeTier - podTier)
		return 0.92 - 0.32*(diff/maxDiff)
	} else {
		// High-tier pod on weaker GPU
		diff := float64(podTier - nodeTier)
		return 0.80 - 0.70*(diff/maxDiff)
	}
}
