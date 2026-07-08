# Planar–Concave Cavity Simulation

Gaussian TEM₀₀ mode simulation of a **planar–concave** (half-symmetric) optical
cavity with:

- Radius of curvature of the concave mirror: **R = 1 m**
- Cavity length: **L = 5 cm**

## Which formula applies

For a two-mirror cavity the g-parameters are `g_i = 1 − L/R_i`. With one planar
mirror (`R₁ = ∞`) and one concave mirror (`R₂ = R`):

```
g1 = 1,   g2 = 1 − L/R = 0.95   →   g1·g2 = 0.95  (stable, since 0 ≤ g1·g2 ≤ 1)
```

The wavefront must be flat at the planar mirror, so **the beam waist sits
exactly on the planar mirror**. This fixes the Rayleigh range from geometry
alone:

```
z_R = √( L (R − L) ) = √(0.05 · 0.95) m = 21.79 cm
```

The wavelength then enters only through the waist:

```
w0        = √( λ z_R / π )                        waist, on the planar mirror
w(z)      = w0 √( 1 + (z/z_R)² )                  envelope inside the cavity
w(mirror) = √( (λ/π) √( L R² / (R − L) ) )        spot on the concave mirror
θ         = λ / (π w0)                            far-field half-angle
```

(For a *symmetric* two-concave-mirror cavity the formula is different —
`z_R = ½√(L(2R − L))` with the waist at the cavity centre — which is the usual
source of the "which formula did you use" discrepancy.)

Mode structure (wavelength-independent):

```
FSR  = c / 2L                       = 3.00 GHz
Δν_T = (FSR/π) arccos(√(g1 g2))     = 215 MHz     transverse mode spacing
ν(q,m,n) = FSR · ( q + (m+n+1)/π · arccos √(g1 g2) )
```

## Results for common laser wavelengths

| λ [nm] | w₀ flat mirror [µm] | w curved mirror [µm] | θ [mrad] | V_mode [mm³] |
|-------:|--------------------:|---------------------:|---------:|-------------:|
|    405 |               167.6 |                172.0 |    0.769 |         1.10 |
|    532 |               192.1 |                197.1 |    0.881 |         1.45 |
|    633 |               209.6 |                215.0 |    0.962 |         1.72 |
|    780 |               232.6 |                238.7 |    1.067 |         2.13 |
|   1064 |               271.7 |                278.7 |    1.247 |         2.90 |
|   1550 |               327.9 |                336.4 |    1.505 |         4.22 |

All spot sizes scale as **√λ**; because `L ≪ z_R` the beam is almost collimated
inside the cavity (the waist grows only ~3 % from the flat to the curved
mirror).

## Running

```bash
pip install numpy matplotlib
python3 cavity_simulation.py
```

Prints the table above and writes `cavity_simulation.png` with four panels:
mode envelope inside the cavity, spot sizes vs. wavelength, divergence vs.
wavelength, and the transverse-mode spectrum within one FSR.

To change geometry or wavelengths, edit the bottom of `cavity_simulation.py`:

```python
cavity = PlanarConcaveCavity(R=1.0, L=0.05)   # metres
wavelengths = [405, 532, 633, 780, 1064, 1550]  # nm
```
