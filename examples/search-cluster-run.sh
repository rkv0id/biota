#!/usr/bin/env bash
# cluster_searches.sh
#
# Eight MAP-Elites searches designed to cover a wide spread of behavioral
# dimensions, signal modes, and border physics on the miniverse cluster.
# Searches run sequentially and are independent -- a failure in one does
# not abort the rest. Each writes into its own archive subdirectory.
#
# Usage:
#   RAY_ADDRESS=10.10.12.1:6379 ./scripts/cluster_searches.sh
#   RAY_ADDRESS=10.10.12.1:6379 ./scripts/cluster_searches.sh 2>&1 | tee searches.log
#
# Requirements: Ray head already running on the cluster.
#   ray start --head --port 6379  (on miniverse-11)

set -euo pipefail

RAY_ADDRESS="${RAY_ADDRESS:-10.10.12.1:6379}"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-3}"
BUDGET="${BUDGET:-400}"
RANDOM_PHASE="${RANDOM_PHASE:-80}"
CALIBRATION="${CALIBRATION:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-archive}"

run_search() {
    local label="$1"
    shift
    echo ""
    echo "========================================================"
    echo "  $label"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"
    biota search "$@" \
        --device "$DEVICE" \
        --batch-size "$BATCH" \
        --workers "$WORKERS" \
        --ray-address "$RAY_ADDRESS" \
        --budget "$BUDGET" \
        --random-phase "$RANDOM_PHASE" \
        --calibration "$CALIBRATION" \
        --preset standard \
        && echo "  ✓ $label done" \
        || echo "  ✗ $label FAILED (continuing)"
}

# 1. Classic morphological baseline -- torus, no signal.
#    The canonical velocity/gyradius/spectral_entropy trio on torus border.
#    Torus removes edge effects so creatures find stable orbits freely.
#    Good reference archive to compare signal vs non-signal dynamics.
run_search "01-torus-morpho" \
    --border torus \
    --descriptors velocity,gyradius,spectral_entropy \
    --output-dir "$OUTPUT_ROOT/01-torus-morpho"

# 2. Movement specialists -- wall, no signal.
#    Targets the space of translational strategies: angular_velocity separates
#    true gliders (low) from orbiters and rotors (high). displacement_ratio
#    separates straight-line movers (high) from circular paths (low).
#    Wall border forces creatures to solve the edge reflection problem,
#    selecting for different movement strategies than torus.
run_search "02-wall-movement" \
    --border wall \
    --descriptors velocity,angular_velocity,displacement_ratio \
    --output-dir "$OUTPUT_ROOT/02-wall-movement"

# 3. Morphological instability -- torus, no signal.
#    Finds creatures that pulse, reshape, or morph over time rather than
#    maintaining a rigid form. morphological_instability = variance of
#    gyradius over the trace tail. activity = mean absolute gyradius change
#    per step. Together they map the space from rigid stable orbiters to
#    wildly reshaping forms. Interesting ecosystem partners for signal runs.
run_search "03-torus-instability" \
    --border torus \
    --descriptors gyradius,morphological_instability,activity \
    --output-dir "$OUTPUT_ROOT/03-torus-instability"

# 4. Signal channel specialists -- wall, signal.
#    The primary signal search. dominant_channel_fraction separates chemical
#    specialists (emit/receive on one channel) from generalists (spread across
#    all 16 channels). Specialists are the best candidates for chemorepulsion
#    experiments -- two specialists using different dominant channels will
#    have near-zero emission-reception compatibility, forcing spatial
#    competition without chemical attraction.
run_search "04-wall-signal-specialist" \
    --border wall \
    --signal-field \
    --descriptors velocity,dominant_channel_fraction,signal_mass_ratio \
    --output-dir "$OUTPUT_ROOT/04-wall-signal-specialist"

# 5. Signal + morphology -- torus, signal.
#    Mixes a morphological axis (gyradius) with two signal axes.
#    signal_field_variance measures how spatially localized the chemical
#    footprint is -- high = signal stays close to the creature, low = signal
#    diffuses widely. Torus border makes signal wrap around the grid,
#    creating a uniform chemical background that shifts the equilibrium.
#    Good source for ecosystem pairs where spatial scale matters.
run_search "05-torus-signal-morpho" \
    --border torus \
    --signal-field \
    --descriptors gyradius,dominant_channel_fraction,signal_field_variance \
    --output-dir "$OUTPUT_ROOT/05-torus-signal-morpho"

# 6. Signal accumulation -- wall, signal.
#    Finds creatures that build large chemical reservoirs vs those that
#    emit and immediately decay. signal_mass_ratio > 1 means signal mass
#    exceeds initial creature mass -- strong chemical presence in the field.
#    High signal_field_variance + high signal_mass_ratio = a creature that
#    floods its local area with concentrated chemistry. Pairs well with
#    creatures from search 04 that have misaligned receptor profiles.
run_search "06-wall-signal-accumulation" \
    --border wall \
    --signal-field \
    --descriptors velocity,signal_mass_ratio,signal_field_variance \
    --output-dir "$OUTPUT_ROOT/06-wall-signal-accumulation"

# 7. Spatial structure -- torus, no signal.
#    Targets the internal mass distribution of creatures rather than their
#    dynamics. growth_gradient measures internal sharpness (labyrinthine
#    vs smooth). spatial_entropy separates compact single-body creatures
#    from diffuse or multi-body forms. Finds morphologically exotic
#    creatures that look interesting in GIFs but may not be fast movers.
run_search "07-torus-spatial" \
    --border torus \
    --descriptors spatial_entropy,growth_gradient,compactness \
    --output-dir "$OUTPUT_ROOT/07-torus-spatial"

# 8. Torus signal specialists -- torus, signal.
#    Same signal descriptor trio as search 04 but on torus border.
#    Periodic boundary changes how signal diffuses: on torus the chemical
#    field wraps around and can accumulate from both sides simultaneously,
#    selecting for different emission/reception strategies than wall.
#    Compare archives 04 and 08 to understand how border physics shapes
#    the signal niche landscape.
run_search "08-torus-signal-specialist" \
    --border torus \
    --signal-field \
    --descriptors velocity,dominant_channel_fraction,signal_mass_ratio \
    --output-dir "$OUTPUT_ROOT/08-torus-signal-specialist"

echo ""
echo "========================================================"
echo "  All searches complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Signal archives (run suggest_signal_pairs.py on these):"
echo "    $OUTPUT_ROOT/04-wall-signal-specialist"
echo "    $OUTPUT_ROOT/05-torus-signal-morpho"
echo "    $OUTPUT_ROOT/06-wall-signal-accumulation"
echo "    $OUTPUT_ROOT/08-torus-signal-specialist"
echo "========================================================"
