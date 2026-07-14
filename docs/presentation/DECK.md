# Acno deck: slide-by-slide redesign

The Canva deck: https://www.canva.com/design/DAHPSt8jrn8/cni9KHsolyAsv6J2eTlDWg/edit

Slide 1 stays exactly as it is. It works because it has almost no words, one warm
color pair (cream paper + chocolate brown), and calm objects (stones, cream swirl,
plant). Every other slide below copies that recipe: one idea per slide, a few words
in the brown banner style, one visual, and everything else spoken out loud from the
speaker notes.

## Style rules for every slide (taken from slide 1)

- Background: the cream grid-paper background already used on all pages. Keep it.
- Headline: white lowercase text in a chocolate brown rectangle, same as slide 1.
- Body text: at most 15 words on the slide. If it needs more words, it goes in the
  speaker notes instead.
- One visual per slide, placed off-center, leaving calm empty space.
- Decorations come from the same family as slide 1. In Canva's Elements search these
  exact terms find matching pieces: "zen stones stack", "cream smear", "green branch",
  "aesthetic plant". Reuse the ones from slide 1 where possible so the deck feels
  like one set of hands made it.
- No new colors. Brown, cream, white, and the green of the plants is the whole palette.

## Slide 2: background (Travis)

Current problem: nine full-sentence bullets and three empty boxes.

On the slide:
- Banner: "why acne happens"
- Three short phrases arranged in a row, each with a small matching element:
  1. "hormones rise" 
  2. "pores clog"
  3. "skin inflames"
- One stat, large, in the white box style: "85% of teens get acne"

Speaker notes:
"Puberty raises hormone levels, which makes skin produce extra oil. That oil clogs
pores, bacteria grow in the clogged pores, and the result is inflammation: pimples.
It mostly shows up on the face, chest and back. Genetics decides a lot of how bad it
gets, stress makes it worse, and for most people it fades after puberty. About 85
percent of people aged 12 to 24 deal with it, and for teens it hits self-esteem at
the worst possible time. Real answers usually need a dermatologist, which is
expensive and slow to book. Most teens just guess."

## Slide 3: dataset (Travis and Matthew)

On the slide:
- Banner: "the data"
- Three white cards, one per dataset, each with just a number and a label:
  - "3,152 faces / skin type"
  - "4,617 photos / acne type"
  - "927 faces / every spot labeled"
- Small line under the cards: "3 public research datasets. No user photos, ever."
- Visual: the class distribution chart (results/class_distribution.png)

Speaker notes:
"We trained on three public Kaggle research datasets: one for skin type with about
three thousand faces labeled dry, normal or oily; one with over four and a half
thousand photos labeled by acne type; and one where dermatology researchers drew a
box around every single acne spot, which is what teaches our severity model. Two
honest problems we found and dealt with: the classes are imbalanced, for example
whiteheads has four times fewer photos than the other types, so we weighted the rare
classes. And these datasets do not document skin tone coverage, which is a known
weakness in dermatology AI. We flag that openly on our limitations. Privacy matters
here: these are research photos with permission. Our own users' photos are never
stored, and never used for training."

## Slide 4: models (Alan and Tanner)

On the slide:
- Banner: "three models, one report"
- Three white cards in a row:
  1. "skin type / dry, normal or oily"
  2. "acne type / which kind of spot"
  3. "spot counter / how severe"
- One line below: "photo in, full skin report out"
- Visual: the phone-scanning-a-face concept (see APPLY.md for options)

Speaker notes:
"One photo goes in and three models look at it. The first says whether skin is dry,
normal or oily. The second identifies the acne type: whiteheads, blackheads, papules,
pustules or cysts, because each type needs different care. The third does not
classify at all, it finds and counts every individual spot, and the count maps to a
severity grade the same way dermatologists grade severity from lesion counts. We did
not train these from scratch: our datasets are small, so we started from MobileNetV2,
a network already trained on 1.4 million images, and fine-tuned it on skin. The spot
counter is YOLOv8-nano, a small fast object detector. All three feed one report."

## Slide 5: results (Alan)

On the slide:
- Banner: "results" (already there)
- Two charts from the training run, framed in the white box style:
  - results/skin_type_confusion_matrix.png
  - results/yolo_sample_detections.png
- One line: the headline accuracy numbers, filled in after the real Colab training
  run, in the form: "skin type: X% / acne type: X% / spot detection: X mAP"

Speaker notes (fill numbers after training):
"On faces the models never saw during training: the confusion matrix on the left
shows where the skin type model is right and where it confuses classes; the diagonal
is correct answers. On the right, the spot detector marks every lesion it found;
the box count gives the severity grade. Say what is honestly weak too: which classes
still get confused and why, class imbalance, small dataset. Judges trust teams who
know their model's limits."

## Slide 6: future application (Tanner)

On the slide:
- Banner: "where this goes"
- Four short phrases, spaced out:
  - "weekly scans, visible progress"
  - "products matched to your skin"
  - "severe cases: see a dermatologist"
  - "next: eczema and rosacea"
- Visual: the rising stones / growth concept (see APPLY.md)

Speaker notes:
"Skin care is a consistency problem, so the biggest future feature is the weekly
re-scan: a severity trend over weeks, so you can actually see improvement instead of
guessing. Product suggestions matched to detected skin type. A telehealth handoff:
severe or cystic findings get a clear nudge to a real dermatologist, because our job
is triage, not treatment. Streaks and reminders keep teens consistent. And the same
scan-and-classify pipeline extends to other teen skin conditions like eczema and
rosacea."

## Slide 7: demo (Matthew)

On the slide:
- Banner: "demo" (already there)
- One screenshot of the real app report page, framed in the white box style
- One line: "scan your face, get your guide"

Speaker notes:
"Walk it live. Upload a photo, and the app finds the face, runs the three models,
and builds the report: skin type, acne type in plain words, severity with the spots
marked on the photo, and a morning and evening routine that non-coders on our team
can edit, because the advice lives in a text file, not in code. Nothing is saved:
the photo is processed in memory and gone when you close the page. Every report ends
with the same line: this is educational guidance, not a diagnosis."

## Slide 8: conclusion (Erwin)

On the slide:
- Banner: "conclusion" (already there)
- Three short lines, generous spacing:
  - "skin answers should not need an appointment"
  - "guidance, never diagnosis"
  - "scan smarter. skin clearer."
- Visual: reuse the stones + plant from slide 1 so the deck ends where it began

Speaker notes:
"The problem was access: real skin guidance is expensive, slow, and awkward for
teens to ask for. Acno gives an instant, private, free first answer. What building
it taught us: the hard part of applied AI was not the model architecture, it was the
data, the class imbalance, the bias risks, and knowing what the model cannot do. We
kept a hard line the whole way: this is guidance, never diagnosis, and severe cases
always get pointed to a dermatologist. Scan smarter, skin clearer."

## Slide 9

Delete it, or keep it as a plain "thank you" in the slide 1 style with the team
names strip.
