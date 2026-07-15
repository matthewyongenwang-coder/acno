# Acno presentation script

Word for word delivery script. Read it out loud during rehearsal, not just in
your head, that is the only way to catch a hiccup before the real thing.

## Speaking order check

Five people, five roles from the planning doc. Here is who talks on which
slide, and how many separate speaking moments each person gets across the
whole presentation:

| Speaker | Slides | Moments |
|---|---|---|
| Erwin | 1 (open), 8 (conclusion), 9 (close) | 3 |
| Travis | 2, 3 (first half) | 2 |
| Matthew | 3 (second half), 7, 7b | 3 |
| Alan | 4 (first half), 5, 5b (first half) | 3 |
| Tanner | 4 (second half), 5b (second half), 6 | 3 |

Before this pass, Erwin only had slide 8, one moment, while Matthew had three
and slide 5b was an unresolved "Alan or Tanner." The fix: Erwin opens with a
short hook before slide 1 even changes, and leads the room into the closing
motto on slide 9, so he bookends the whole talk instead of only showing up
once. Slide 5b got split in half between Alan and Tanner instead of going to
just one of them. Everyone now has two or three moments, and nobody is
standing there with nothing to say.

The order itself, background, data, models, results, what broke, future, demo,
design, conclusion, already builds a strong arc: state the problem, show the
work is honest, prove it works, own what failed, point forward, prove it again
live, explain why it feels the way it does, close on the message. Keep that
order.

## Say it right (read this once before you rehearse)

- MobileNetV2: say "Mobile Net vee two"
- YOLOv8-nano: say "yolo vee eight nano"
- mAP: say "em A P," or just say the full phrase, "mean average precision"
- epoch: say "EE-pock"
- cystic: say "SIS-tik"

None of these need to be said perfectly, just said the same way every time so
it does not sound like a guess.

## Timing

Roughly fifteen minutes read at a normal, unhurried pace. If your slot is
shorter, cut from slide 5b first (it is the most skippable under time
pressure), then trim slide 2's stat line and slide 6's fourth bullet.

---

## Slide 1: title

No slide change yet. This line is said into the silence before the deck even
moves, so the room has something to listen to right away.

**ERWIN:**
"Getting real advice about your skin usually means booking a dermatologist.
That takes weeks, it costs money, and for a teenager who is already
self-conscious about their face, it is the last thing they want to do.

We built Acno so you do not have to wait for any of that."

[ADVANCE SLIDE]

*"To understand why we built this, here is Travis on what is actually
happening on your skin."*

---

## Slide 2: background

**TRAVIS:**
"Puberty raises your hormone levels.

Higher hormones mean your skin makes more oil. That oil clogs your pores.
Bacteria grow inside the clogged pore. The result is inflammation, a pimple.

It shows up mostly on the face, chest, and back. Genetics decides a lot of how
bad it gets. Stress makes it worse. For most people, it fades after puberty.

About eighty five percent of teens deal with acne at some point. And it hits
self-esteem at the exact age when self-esteem is already fragile.

Real answers usually mean a dermatologist visit. That is expensive, and it is
slow to book. So most teens just guess."

[ADVANCE SLIDE]

*"Guessing is not good enough. So we built something that does not guess, it
learns. Here is what we trained it on."*

---

## Slide 3: dataset (SPLIT: Travis first half, then Matthew second half)

**TRAVIS:**
"We trained on three public research datasets from Kaggle.

One has about three thousand faces, labeled dry, normal, or oily.

One has over four thousand six hundred photos, labeled by acne type.

And one is special. Dermatology researchers drew a box around every single
spot on nine hundred twenty seven faces. That one teaches our model to count
severity."

*"Matthew, tell them the part we are not going to hide."*

**MATTHEW:**
"Two things we are honest about.

First, the classes are not balanced. Whiteheads has four times fewer photos
than the other types, so we weighted the rare classes higher during training.

Second, none of these datasets document skin tone coverage. That is a known
weakness in dermatology AI, and we are not going to pretend it is not. We flag
it as a limitation, not hide it.

And on privacy: these training photos are public research data, used with
permission. Your photo, the one you scan with Acno, is never stored, and never
used to train anything."

[ADVANCE SLIDE]

*"So that is what the models learned from. Here are Alan and Tanner on what
the models actually do with it."*

---

## Slide 4: models (SPLIT: Alan first half, then Tanner second half)

**ALAN:**
"One photo goes in. Three models look at it.

The first model checks your skin type: dry, normal, or oily.

The second identifies the acne type, whiteheads, blackheads, papules,
pustules, or cysts, because each one needs different care."

*"Tanner, the third one is yours."*

**TANNER:**
"The third model does not classify anything. It finds and counts every single
spot on your face. That count becomes a severity grade, the same way real
dermatologists grade severity by counting lesions.

We did not train these from scratch. Our datasets are small, so we started
from MobileNetV2, a network already trained on one point four million images,
and fine-tuned it on skin. The spot counter runs on YOLOv8-nano, a small, fast
object detector.

All three feed into one report."

[ADVANCE SLIDE]

*"So does it actually work? Alan has the numbers."*

---

## Slide 5: results

**ALAN:**
"These results are on faces the models never saw during training.

This chart is the acne model learning. Accuracy climbs, loss falls, epoch by
epoch. That kink in the middle is where we unfroze the backbone and let it
fine-tune.

It landed at fifty nine percent accuracy across five acne types. That is about
three times better than random guessing. And the rarest class, whiteheads,
gets caught ninety eight percent of the time, thanks to that class weighting
Matthew mentioned earlier.

On the right, the spot detector marks every lesion it finds, with a mean
average precision of point six seven. That box count is what drives the
severity grade.

Now, the honest one. Skin type only reached forty three percent. That
dataset's labels are subjective, the model overfit, and honestly, that result
taught us more than any of our wins did."

[ADVANCE SLIDE]

*"Which brings us to what actually went wrong."*

---

## Slide 5b: what broke (SPLIT: Alan first half, then Tanner second half)

**ALAN:**
"Three real problems. Three fixes.

First: class imbalance. Whiteheads had four times fewer photos than the other
acne types, so early versions of our model just ignored them. We fixed it with
class weights, a mistake on a rare class now costs the model more during
training.

Second: overfitting. This graph is our skin type model memorizing instead of
learning. The blue training line climbs to sixty four percent. The orange
validation line stays flat around thirty eight. The graph told us the problem
was the data, not the model."

*"Tanner, tell them about the day everything just broke."*

**TANNER:**
"Third: the day we built the app, a library update deleted a function our face
detector depended on. Everything broke, with no warning.

One line, pinning the older library version, fixed it. It also taught us why
real projects lock their dependency versions before it is too late.

Debugging was half this project. Honestly, it was half the learning too."

[ADVANCE SLIDE]

*"Tanner, take us to where this actually goes."*

---

## Slide 6: future

**TANNER:**
"Skin care is a consistency problem, not a knowledge problem. So the biggest
feature we want next is a weekly re-scan, a severity trend over time, so you
can actually see progress instead of guessing.

Next, product suggestions matched to your specific skin type.

For severe or cystic cases, a clear nudge toward a real dermatologist, because
our job is triage, not treatment.

Streaks and reminders, to keep teens actually consistent.

And the same scan-and-classify pipeline can extend to other teen skin
conditions, eczema, rosacea."

[ADVANCE SLIDE]

*"Enough talking about it. Matthew is going to show you."*

---

## Slide 7: demo

**MATTHEW:**
"I am going to run this live.

[Open the app and walk through it on screen]

Upload a photo. The app finds the face, and runs all three models.

It builds a full report: skin type, acne type in plain words, severity with
every spot marked directly on the photo, and a morning and evening routine.

That routine text lives in a plain file, not in code, so anyone on our team
can edit the advice without touching a single line of code.

Nothing is saved. The photo is processed in memory, and it is gone the moment
you close the page.

And every single report ends with the same line: this is educational guidance.
Not a diagnosis."

[ADVANCE SLIDE]

*"So it works. But we also cared a lot about how it feels to use. That part is
mine too."*

---

## Slide 7b: design

**MATTHEW:**
"Acne already comes with enough shame. A lot of teens feel like asking for
help means walking into a cold, clinical waiting room. So we designed against
that, on purpose.

The background is warm cream, not hospital white.

The accent color is a muted sage green, the color of plants and skin, not the
alert red most health apps default to.

Headlines are set in a soft serif font. It reads more like a wellness journal
than a lab report.

Every card has rounded corners and a soft shadow, instead of the hard
rectangles you would see on a medical form.

And we were deliberate about red. The only red anywhere in this app is the box
drawn around a spot on your own photo. That is it. So it stays meaningful,
instead of turning the whole screen into a warning sign.

Even our most serious message, see a dermatologist, sits in a soft peach box,
not a red banner. Because we wanted it to read as care. Not an alarm.

We wanted an app that feels like it is on your side, before it has told you
anything about your skin at all."

[ADVANCE SLIDE]

*"And that is really the whole point of this project. Erwin is going to close
us out."*

---

## Slide 8: conclusion

**ERWIN:**
"The problem here was access. Real skin guidance is expensive, it is slow, and
for a teenager, it is awkward to even ask for.

Acno gives you an instant, private, first answer, for free.

Here is what building it actually taught us: the hard part of applied AI was
never the model architecture. It was the data. It was the class imbalance. It
was the bias risks. It was knowing exactly what our model cannot tell you.

We held one line the entire way through this project: guidance, never
diagnosis. Every severe case still gets pointed to a real dermatologist."

[ADVANCE SLIDE]

---

## Slide 9: thank you

**ERWIN, solo:**
"Thank you."

**EVERYONE, together:**
"Scan smarter. Skin clearer."

[HOLD, then open the floor to questions]
