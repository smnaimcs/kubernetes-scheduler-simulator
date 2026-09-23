package plugin

import (
	"testing"

	simontype "github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/type"
	"github.com/stretchr/testify/assert"
)

func TestCalculateTierBinPackingScore(t *testing.T) {
	// (a) GPU pod onto empty GPU node (perfect match)
	t.Run("perfect match - unchanged score", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-empty",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     10000,
			GpuNumber:        4,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  2000,
			GpuNumber: 2,
			MilliGpu:  1000,
			GpuType:   "V100M16",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(39), score)
	})

	// (b) Low-tier pod on stronger GPU (pod T4, node A10)
	t.Run("low-tier pod on stronger GPU (pod T4, node A10)", func(t *testing.T) {
		podTier, podKnown := simontype.GpuTypeToTier("T4")
		nodeTier, nodeKnown := simontype.GpuTypeToTier("A10")
		m := simontype.TierMultiplier(podTier, nodeTier, podKnown, nodeKnown)
		assert.True(t, m < 0.92)
		assert.True(t, m >= 0.60)

		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-a10",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     10000,
			GpuNumber:        4,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000},
			GpuType:          "A10",
		}
		podRes := simontype.PodResource{
			MilliCpu:  2000,
			GpuNumber: 2,
			MilliGpu:  1000,
			GpuType:   "T4",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(25), score)
	})

	// (c) High-tier pod on weaker GPU (pod A10, node T4)
	t.Run("high-tier pod on weaker GPU (pod A10, node T4)", func(t *testing.T) {
		podTier, podKnown := simontype.GpuTypeToTier("A10")
		nodeTier, nodeKnown := simontype.GpuTypeToTier("T4")
		m := simontype.TierMultiplier(podTier, nodeTier, podKnown, nodeKnown)
		assert.True(t, m >= 0.10 && m <= 0.80)

		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-t4",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     10000,
			GpuNumber:        4,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000},
			GpuType:          "T4",
		}
		podRes := simontype.PodResource{
			MilliCpu:  2000,
			GpuNumber: 2,
			MilliGpu:  1000,
			GpuType:   "A10",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(8), score)
	})

	// (d) Stage 3 ceiling penalty: s_cpu exactly 0.85 -> no penalty
	t.Run("utilisation ceiling: s_cpu exactly 0.85 (no penalty)", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     8000,
			GpuNumber:        4,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  6500,
			GpuNumber: 0,
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(15), score)
	})

	// (e) Stage 3 ceiling penalty: s_cpu 0.86 -> penalty applied
	t.Run("utilisation ceiling: s_cpu 0.86 (penalty applied)", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     8000,
			GpuNumber:        4,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  6600,
			GpuNumber: 0,
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(6), score)
	})

	// (f) Stage 3 ceiling penalty: s_gpu 0.90 -> penalty applied
	t.Run("utilisation ceiling: s_gpu 0.90 (penalty applied)", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-10gpus",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     8000,
			GpuNumber:        10,
			MilliGpuLeftList: []int64{0, 0, 0, 0, 0, 0, 0, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  1000,
			GpuNumber: 2,
			MilliGpu:  1000,
			GpuType:   "V100M16",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(27), score)
	})

	// (g) Stage 4 Look-ahead anti-fragmentation: r = 0 (1.15x multiplier)
	// Node 4 GPUs free, Pod 4 GPUs -> r = 0 -> s_gpu = 1.0 (>0.85 -> 0.40 ceiling penalty applies)
	// base = 0.70 -> with 0.40 ceiling penalty = 0.28 -> with 1.15 lookahead = 0.322 -> score = 32
	t.Run("look-ahead anti-fragmentation: r = 0 (1.15x multiplier)", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-4gpus",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     10000,
			GpuNumber:        4,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  2000,
			GpuNumber: 4,
			MilliGpu:  1000,
			GpuType:   "V100M16",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(32), score)
	})

	// (h) Stage 4 Look-ahead anti-fragmentation: r = 4 (1.10x multiplier)
	// Node 8 GPUs free, Pod 4 GPUs -> r = 4 (clean multiple of 4) -> multiplier 1.10
	// base = 0.3875 -> final = 0.3875 * 1.10 = 0.42625 -> score = 43
	t.Run("look-ahead anti-fragmentation: r = 4 (1.10x multiplier)", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-8gpus",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     10000,
			GpuNumber:        8,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  2000,
			GpuNumber: 4,
			MilliGpu:  1000,
			GpuType:   "V100M16",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(43), score)
	})

	// (i) Stage 4 Look-ahead anti-fragmentation: r = 5 (0.90x penalty multiplier)
	// Node 9 GPUs free, Pod 4 GPUs -> r = 5 (r % 4 != 0) -> multiplier 0.90
	// base = 0.352777... -> final = 0.352777... * 0.90 = 0.3175 -> score = 32
	t.Run("look-ahead anti-fragmentation: r = 5 (0.90x penalty)", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-9gpus",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     10000,
			GpuNumber:        9,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  2000,
			GpuNumber: 4,
			MilliGpu:  1000,
			GpuType:   "V100M16",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(32), score)
	})

	// (j) Stage 4 Look-ahead anti-fragmentation: r = 3 (0.90x penalty multiplier)
	// Node 7 GPUs free, Pod 4 GPUs -> r = 3 (r % 4 != 0) -> multiplier 0.90
	// base = 0.4321428... -> final = 0.4321428... * 0.90 = 0.388928... -> score = 39
	t.Run("look-ahead anti-fragmentation: r = 3 (0.90x penalty)", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-7gpus",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     10000,
			GpuNumber:        7,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  2000,
			GpuNumber: 4,
			MilliGpu:  1000,
			GpuType:   "V100M16",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(39), score)
	})

	// (k) Stage 4 Look-ahead disabled for pod.GpuNumber = 3 (pod.GpuNumber < 4)
	// Node 7 GPUs free, Pod 3 GPUs -> r = 4, but pod.GpuNumber = 3 < 4 so multiplier 1.0 (no look-ahead)
	// base = 0.342857... -> score = 34
	t.Run("look-ahead disabled for pod.GpuNumber = 3 (no multiplier)", func(t *testing.T) {
		nodeRes := simontype.NodeResource{
			NodeName:         "gpu-node-7gpus",
			MilliCpuCapacity: 10000,
			MilliCpuLeft:     10000,
			GpuNumber:        7,
			MilliGpuLeftList: []int64{1000, 1000, 1000, 1000, 1000, 1000, 1000},
			GpuType:          "V100M16",
		}
		podRes := simontype.PodResource{
			MilliCpu:  2000,
			GpuNumber: 3,
			MilliGpu:  1000,
			GpuType:   "V100M16",
		}
		score := calculateTierBinPackingScore(nodeRes, podRes)
		assert.Equal(t, int64(34), score)
	})
}
