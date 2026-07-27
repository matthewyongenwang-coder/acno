# Fairness review: does Acno work equally well across skin tones?

Written July 2026. This is the first milestone of the funded project plan and the
biggest known gap in the project. Reproduce with:

```bash
.venv/bin/python scripts/estimate_skin_tone.py
.venv/bin/python scripts/validate_tone_estimate.py
.venv/bin/python scripts/fairness_eval.py
```

## The short answer

**On acne_type we found no meaningful gap, and we had enough data to have seen one.**
**On skin_type and the lesion detector we found no gap, but we could not have seen one
even if it were large.** And across all three datasets, the deeper problem is not the
model but the data: there is almost no dark skin in any of them, so the question people
most want answered cannot be answered with what we have.

Nothing here licenses the claim "Acno works equally well for everyone." It does not
support that claim, and we should not make it.

## Why this needed estimating at all

None of the three Kaggle datasets record skin tone. To measure performance by tone we
first had to assign a tone to every image, which means estimating it from pixels.

We use the Individual Typology Angle (ITA), the standard proxy in dermatology imaging:

    ITA = arctan((L* - 50) / b*) * 180 / pi

computed in CIELAB over pixels identified as skin, taking the median after discarding
the darkest and brightest 20% so that shadows and shiny foreheads do not dominate.

## Two ways we got this wrong before getting it right

Both errors were caught by looking at the images rather than the numbers, which is why
the scripts write contact sheets and why nobody should trust a tone number they have not
eyeballed.

**First attempt: the formula broke on pink images.** We used `arctan2(L - 50, b)`, which
silently accepts a negative or near-zero b\*. Real skin always has a positive b\* because
it is yellowish. Pinkish close-ups and red annotation arrows have b\* near zero, and they
landed in the "dark" bin. The fix is the standard `arctan` form plus a requirement that
b\* be positive, and images that fail it are marked unknown rather than guessed at.

**Second attempt: backgrounds contaminated skin_type.** The skin_type images are
composed marketing photographs, and a pink or orange backdrop passes a YCrCb skin test
happily. Measured over the whole frame, a Black woman photographed against a pink
background was placed in the **lightest** quartile, and light-skinned people against warm
backgrounds were placed in the **darkest**. The estimate was close to meaningless. The
fix is to measure only inside the detected face region, which is what
`scripts/build_face_crops.py` already produces. This works, at the cost of covering only
the 62% of skin_type images where a face is actually detectable.

acne_type images are close-ups that are almost entirely skin, so they are measured on the
whole frame and 99.6% get a usable estimate.

## How we checked the estimate before trusting it

**Label independence.** If estimated tone tracked the class label, any "fairness gap"
would just be the label in disguise. This was a real worry for acne_type, whose images
are close-ups where inflamed skin is red. Measured as the spread of mean ITA across
classes divided by the spread across images:

| Dataset | Ratio | Verdict |
|---|---|---|
| skin_type | 4.8% | tone estimate is essentially independent of the label |
| acne_type | 9.2% | mild association, small enough to proceed |

**Visual checks.** Contact sheets of each band and of images sorted by ITA, inspected by
eye. After the face-region fix, the darkest skin_type band contains genuinely dark-skinned
faces and the lightest contains pale ones.

## Groups are quartiles, not Fitzpatrick types

ITA gets the *ordering* of skin tone roughly right on these photos. Its *absolute value*
does not survive uncontrolled lighting, white balance and filters. So we do not claim
"5% of our data is Fitzpatrick VI." We split at the quartiles of each training
distribution and talk about "the darkest-appearing quarter of our data," which the
measurement can actually support.

This matters when reading the results below: **the darkest quarter of these datasets is
not dark skin.** It is the darker end of a range that runs from pale to roughly tan.

## Representation: the real finding

| Dataset | Usable estimates | Share in the darkest ITA band |
|---|---|---|
| skin_type (face region) | 61.8% of training images | 2.6% of training, **0 images in the test split** |
| acne_type | 99.6% of training images | 1.6% of training |

The skin_type test split contains **no images at all** in the darkest absolute band. That
single fact caps what any evaluation on it can tell us.

## Results by tone group

Measured on the strict, leak-free test split, with test-time augmentation, using the
shipped models.

### acne_type (n = 321)

Two models, before and after adding tone-aware augmentation:

| Group | n | Before (`medium` aug) | After (`tone` aug) |
|---|---|---|---|
| lightest quarter | 85 | 96.5% | **97.6%** |
| second | 65 | 98.5% | **100.0%** |
| third | 80 | 100.0% | **100.0%** |
| darkest quarter | 91 | 95.6% | **97.8%** |
| spread best to worst | | 4.4 points | **2.4 points** |
| overall (strict) | 321 | 97.5% | **98.8%** |

Training with hue, saturation and brightness jitter across the range that separates
lighter from darker skin **raised every group and narrowed the spread**, with the
darkest quarter gaining the most (+2.2 points). That is the version now shipping. It is
the one concrete fairness improvement in this review, and it cost nothing but a
different augmentation setting.

### skin_type (n = 114, of which 44 have no usable tone estimate)

| Group | n | Accuracy | 95% CI | Macro F1 |
|---|---|---|---|---|
| lightest quarter | 21 | 52.4% | [32.4%, 71.7%] | 0.402 |
| second | 19 | 36.8% | [19.1%, 59.0%] | 0.377 |
| third | 20 | 50.0% | [29.9%, 70.1%] | 0.417 |
| darkest quarter | 10 | 50.0% | [23.7%, 76.3%] | 0.376 |

Spread: 15.5 points, but with 10 images in the smallest group this is noise.

### Lesion detector (n = 48, split at the median into two halves)

| Group | n | mAP50 | Precision | Recall |
|---|---|---|---|---|
| lighter half | 24 | 0.691 | 0.740 | 0.589 |
| darker half | 24 | 0.639 | 0.622 | 0.605 |

Gap: 0.051 mAP50.

## The most important number in this document

How large a gap would each comparison have had to be for us to detect it reliably
(two-proportion test, 80% power, alpha 0.05)?

| Comparison | Group sizes | Smallest detectable gap |
|---|---|---|
| acne_type lightest vs darkest | 85 vs 91 | **7.2 points** |
| skin_type lightest vs darkest | 21 vs 10 | **53.7 points** |
| detector lighter vs darker | 24 vs 24 | **38.0 points** |

So "no significant difference" means genuinely different things per model:

- **acne_type**: a real result. A gap larger than about 7 points would very likely have
  shown up, and it did not. Within the tone range this dataset covers, the model looks
  even-handed.
- **skin_type**: not a result. A 40-point gap could be sitting there undetected.
- **lesion detector**: not a result, for the same reason. The 5-point gap we measured is
  well inside noise, and a 30-point gap would also have been.

## What we will not claim

- We will not say Acno works equally well on dark skin. We have barely tested it on dark
  skin, because there is barely any in the data.
- We will not quote the skin_type or detector fairness numbers as evidence of fairness.
  They are evidence of a small test set.
- We will not translate ITA into Fitzpatrick types for individual people.

## What would actually settle this

1. **A tone-diverse evaluation set.** A few hundred images spanning the full range, with
   tone recorded rather than estimated, would move the detectable gap for skin_type from
   50 points to under 10. This is the single highest-value thing left, and it is the
   natural thing to ask Jocelyn's clinic dermatologists about.
2. **Tone-aware training, already done for acne_type.** `--augment tone` improved every
   group and narrowed the spread, so it ships. Applied to skin_type it changed nothing
   measurable (44.7% strict against a seed mean of 44.4%), which is what you would
   expect from a model whose limit is label quality rather than colour sensitivity. It
   is worth re-testing on skin_type once there is a better dataset.
3. **Report per-group scores every time we retrain**, not once. `scripts/fairness_eval.py`
   is cheap to re-run and should be part of the routine, the same way we re-run the
   leak scan.

## For the grant report

The funded milestone was "complete a fairness review of the models across a range of skin
tones, and document the results honestly, by October 2026." This is that review,
delivered early. The honest version of the finding is:

> We built a skin tone estimator, validated it, found and fixed two ways it was wrong,
> and measured all three models across tone groups. The acne classifier shows no
> meaningful difference across the tone range present in its data. For the other two
> models our test sets are too small to support any conclusion, and we say so rather than
> reporting a reassuring number we cannot stand behind. The limiting factor throughout is
> that these public datasets contain very little dark skin, which is exactly the gap the
> dermatology AI literature warns about, and it is now measured rather than assumed.
