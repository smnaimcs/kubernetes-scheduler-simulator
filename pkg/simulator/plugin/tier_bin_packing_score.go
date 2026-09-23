package plugin

import (
	"context"
	"fmt"
	"math"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"

	simontype "github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/type"
	gpushareutils "github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/type/open-gpu-share/utils"
	"github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/utils"
)

/*
=============================================================================
Tier-Matched Bin-Packing with Look-Ahead Anti-Fragmentation
=============================================================================

Reference paper: "Mitigating GPU fragmentation in heterogeneous Kubernetes
clusters via tier-matched bin-packing with look-ahead anti-fragmentation"
(FGCS-D-26-03340).

Design goal: a drop-in sibling of FGDScorePlugin that is a pure ScorePlugin.
The Filter (predicate) phase is unchanged. Only Score and the post-Score
GPU-ID allocation (via allocateGpuIdFunc) are extended.

Key departures from FGD (USENIX ATC 2023):
  | FGD                                    | This plugin                          |
  |----------------------------------------|--------------------------------------|
  | Fractional GPU sharing                 | Exclusive whole-GPU allocation       |
  | GPU type as hard filter                | Soft tier matching (graduated)       |
  | Continuous gradient minimisation       | Integer remainder + weighted packing |
  | No burst-headroom protection           | 85% utilisation ceiling (soft)       |
  | Complex per-node gradient              | O(1) per-node arithmetic             |

-----------------------------------------------------------------------------
Stage 0. Filter phase (UNCHANGED, owned by other plugins)
-----------------------------------------------------------------------------

NodeResourcesFit, NodeUnschedulable, PodCount, NodeAffinity/NodeSelector,
TaintToleration, and Open-Gpu-Share all still run before this plugin's Score
is invoked. If any of them reject the node, Score is never called.

-----------------------------------------------------------------------------
Stage 1. Base score (weighted bin-packing consolidation)
-----------------------------------------------------------------------------

Post-placement utilisation fractions:

  s_gpu = (n.GPU_used    + p.GPU_req)    / n.GPU_total
  s_cpu = (n.CPU_used    + p.CPU_req)    / n.CPU_total
  s_mem = (n.Memory_used + p.Memory_req) / n.Memory_total

  NOTE: Memory fields are currently commented out of simontype.PodResource
  and simontype.NodeResource in this repository. In this implementation the
  memory term is DROPPED and the remaining weights are renormalised:
      0.50 / 0.30 / 0.20  ->  0.625 / 0.375  (GPU / CPU)
  This preserves the relative GPU:CPU importance of the original design.
  If memory is later added to the type system, restore the 0.50/0.30/0.20
  weights verbatim and remove this note.

For GPU-intensive pods (p.GpuNumber > 0):

  base_score = 0.625 * s_gpu + 0.375 * s_cpu

For CPU-only pods (p.GpuNumber == 0):

  base_score = LeastAllocated(s_cpu, s_mem)   // spread-first, as in default
                                              // kube-scheduler; memory term
                                              // omitted per note above
  if n.GpuNumber == 0:
      base_score = base_score * 1.5           // soft bonus for CPU-only node

The 1.5x bonus is a SOFT steering signal, not a hard quarantine: if all
CPU-only nodes are full, CPU pods may still land on GPU nodes.

-----------------------------------------------------------------------------
Stage 2. GPU tier matching (soft multiplier)
-----------------------------------------------------------------------------

Tier ordering (highest to lowest):
    A10 > G3 > V100M32 > V100M16 > G2 > T4 > P100

(See pkg/type/gpu_tier.go for the authoritative mapping; the tier table is
derived from the GPU model strings that actually appear in this repo's
data/ traces, and any type not present there is treated as "unknown" with
multiplier 1.0.)

  diff    = |podTier - nodeTier|
  maxDiff = (number of distinct tiers) - 1

  perfect match (diff == 0):                m_tier = 1.00
  low-tier pod on stronger GPU:             m_tier = 0.92 - 0.32*(diff/maxDiff)
  high-tier pod on weaker GPU:              m_tier = 0.80 - 0.70*(diff/maxDiff)
  either side unknown:                      m_tier = 1.00  (no penalty)

  score = base_score * m_tier

The heavier penalty for "high-tier pod on weak GPU" reflects the real
performance degradation risk; the lighter penalty for "low-tier pod on
strong GPU" reflects wasted premium capacity.

This is a SOFT multiplier. It never causes a node to be rejected. If no
tier-matched node exists, the pod is scheduled anyway with a penalty.

-----------------------------------------------------------------------------
Stage 3. Utilisation ceiling (soft penalty)
-----------------------------------------------------------------------------

  if max(s_gpu, s_cpu, s_mem) > 0.85:
      score = score * 0.40

Rationale: pack consolidation is desirable, but leaving zero headroom causes
burst arrivals to be unschedulable. 85% / 0.40 were chosen empirically from
the Alibaba OpenB trace as the balance between consolidation and headroom.

This is a penalty, NOT a filter. Nodes above 85% remain eligible.

-----------------------------------------------------------------------------
Stage 4. Look-ahead anti-fragmentation (activated only for p.GpuNumber >= 4)
-----------------------------------------------------------------------------

Let:
    r = number of free GPUs on the node after placement
      = freeGpus(n) - p.GpuNumber

  if r == 0:              score = score * 1.15   // perfect fit
  elif r % 4 == 0:        score = score * 1.10   // clean multiple of 4
  elif r % 4 != 0:        score = score * 0.90   // stranded 1-3 GPU slot

Rationale: 4-GPU jobs are the coarsest common shape in the OpenB trace;
leaving 1-3 GPUs free on a node permanently strands them for future 4-GPU
jobs. Perfect fits and clean multiples of 4 preserve large-job capacity.

The threshold GpuNumber >= 4 is chosen because 1-3 GPU requests are common
and do not cause severe stranding on their own. Smaller pods receive no
look-ahead adjustment (multiplier 1.0).

-----------------------------------------------------------------------------
Stage 5. Final score and selection
-----------------------------------------------------------------------------

  score_n = base_score
          * m_tier
          * (0.40 if over 85% ceiling else 1.0)
          * lookahead_multiplier

The result is scaled to [framework.MinNodeScore, framework.MaxNodeScore]
(0..100) and returned from Score(). The framework itself picks the argmax
across eligible nodes. Deterministic lexicographic tie-breaking is NOT
achievable from inside a ScorePlugin in the k8s scheduling framework; see
the note in the Score() doc below.

-----------------------------------------------------------------------------
Stage 6. Bind / GPU-ID allocation
-----------------------------------------------------------------------------

allocateGpuIdBasedOnTierBinPacking() is registered into the global
allocateGpuIdFunc map by NewTierBinPackingScorePlugin and is called during
the reserve/bind stage to commit the specific GPU device indices for the
pod. This plugin uses simontype.AllocateExclusiveGpuId because the
algorithm assumes whole-GPU exclusive allocation.

-----------------------------------------------------------------------------
Pre-computed constants (parameter justification, from Section 6 of paper)
-----------------------------------------------------------------------------

  GPU / CPU weights (renormalised):   0.625 / 0.375
  Utilisation ceiling:                0.85
  Ceiling penalty multiplier:         0.40
  Tier multiplier range:              0.10 .. 0.92 (see gpu_tier.go)
  Look-ahead multipliers:             1.15 / 1.10 / 0.90
  Look-ahead activation threshold:    p.GpuNumber >= 4

=============================================================================
*/

type TierBinPackingScorePlugin struct {
	handle framework.Handle
}

var _ framework.ScorePlugin = &TierBinPackingScorePlugin{}

func NewTierBinPackingScorePlugin(_ runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	plugin := &TierBinPackingScorePlugin{
		handle: handle,
	}
	allocateGpuIdFunc[plugin.Name()] = allocateGpuIdBasedOnTierBinPacking
	return plugin, nil
}

func (plugin *TierBinPackingScorePlugin) Name() string {
	return simontype.TierBinPackingScorePluginName
}

func (plugin *TierBinPackingScorePlugin) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeResPtr := utils.GetNodeResourceViaHandleAndName(plugin.handle, nodeName)
	if nodeResPtr == nil {
		return framework.MinNodeScore, framework.NewStatus(framework.Error, fmt.Sprintf("failed to get nodeRes(%s)\n", nodeName))
	}
	nodeRes := *nodeResPtr
	podRes := utils.GetPodResource(p)

	score := calculateTierBinPackingScore(nodeRes, podRes)
	return score, framework.NewStatus(framework.Success)
}

func (plugin *TierBinPackingScorePlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func numFreeGpus(nodeRes simontype.NodeResource) int {
	free := 0
	for _, gpuMilliLeft := range nodeRes.MilliGpuLeftList {
		if gpuMilliLeft == gpushareutils.MILLI {
			free++
		}
	}
	return free
}

func calculateTierBinPackingScore(nodeRes simontype.NodeResource, podRes simontype.PodResource) int64 {
	var sGpu float64
	if nodeRes.GpuNumber > 0 {
		usedGpus := float64(nodeRes.GpuNumber - numFreeGpus(nodeRes))
		sGpu = (usedGpus + float64(podRes.GpuNumber)) / float64(nodeRes.GpuNumber)
	} else {
		sGpu = 0.0
	}

	var sCpu float64
	if nodeRes.MilliCpuCapacity > 0 {
		usedCpu := float64(nodeRes.MilliCpuCapacity - nodeRes.MilliCpuLeft)
		sCpu = (usedCpu + float64(podRes.MilliCpu)) / float64(nodeRes.MilliCpuCapacity)
	} else {
		sCpu = 0.0
	}

	var base float64
	if podRes.GpuNumber > 0 {
		// GPU-intensive pods: weighted bin-packing consolidation.
		// Memory fields (0.20 weight) are currently commented out of simontype.PodResource
		// and simontype.NodeResource in this repository. Dropping the memory term
		// renormalizes remaining weights: 0.50/0.80 = 0.625 (GPU) and 0.30/0.80 = 0.375 (CPU).
		base = 0.625*sGpu + 0.375*sCpu
	} else {
		// CPU-only pods: least-allocated (spread-first) using (1 - sCpu).
		// Memory term omitted per type system limitation note above.
		base = 1.0 - sCpu
		if nodeRes.GpuNumber == 0 {
			base = base * 1.5 // Soft bonus for CPU-only node
		}
	}

	// Stage 2. GPU tier matching (soft multiplier)
	podTier, podKnown := simontype.GpuTypeToTier(podRes.GpuType)
	nodeTier, nodeKnown := simontype.GpuTypeToTier(nodeRes.GpuType)
	mTier := simontype.TierMultiplier(podTier, nodeTier, podKnown, nodeKnown)

	finalScoreVal := base * mTier

	// Stage 3. Utilisation ceiling (soft penalty)
	if math.Max(sGpu, sCpu) > 0.85 {
		finalScoreVal *= 0.40
	}

	// Stage 4. Look-ahead anti-fragmentation (activated only for p.GpuNumber >= 4)
	if podRes.GpuNumber >= 4 {
		freeGpus := numFreeGpus(nodeRes)
		r := freeGpus - podRes.GpuNumber
		if r < 0 {
			// Negative r means the pod doesn't fit; the framework filter should
			// have prevented it, but guard by returning MinNodeScore.
			return framework.MinNodeScore
		}
		if r == 0 {
			finalScoreVal *= 1.15
		} else if r%4 == 0 {
			finalScoreVal *= 1.10
		} else {
			finalScoreVal *= 0.90
		}
	}

	score := int64(math.Round(finalScoreVal * float64(framework.MaxNodeScore)))
	if score > framework.MaxNodeScore {
		score = framework.MaxNodeScore
	}
	if score < framework.MinNodeScore {
		score = framework.MinNodeScore
	}
	return score
}

func allocateGpuIdBasedOnTierBinPacking(nodeRes simontype.NodeResource, podRes simontype.PodResource, _ simontype.GpuPluginCfg, _ *simontype.TargetPodList) (gpuId string) {
	return simontype.AllocateExclusiveGpuId(nodeRes, podRes)
}
