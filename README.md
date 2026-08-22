# NNI-Remake
A remake of the archived NNI repository by Microsoft. 

## Working mode: rapid prototyping

Training runs are screening tools, not marathons. Default budget is **40
epochs per run (~10-15 min on the RTX 4060)**; never leave a run going 5+
hours without a specific reason. Workflow per task family:

1. Screen every architecture variant at 40 epochs (same recipe, cosine scaled
   to the short schedule).
2. Only variants within striking distance of the accuracy target get one
   longer confirmation run (60-200 epochs).

Longer runs must be justified up front (e.g.
`Tres-cifar-microres-bnlong-M11` at 60 epochs tests whether schedule length
closes the last accuracy gap).