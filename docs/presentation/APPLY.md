# How to apply the deck redesign

Read DECK.md first. It has the exact text for every slide and the speaker notes.
This file is the checklist for actually doing it in Canva.

## Before you start

1. Do not touch slide 1. It is the style reference for everything else.
2. Run the training notebook on Colab first if you want the real charts
   (notebooks/Acno.ipynb, Runtime > Run all on a GPU). It downloads a
   results zip with every chart the deck needs. Until then you can lay out
   the slides and leave chart placeholders.

## Who does what

| Slide | Owner | What to do |
|---|---|---|
| 2 background | Travis | Replace the 9 bullets with the 3 phrases + 1 stat from DECK.md. Move the sentences into speaker notes. |
| 3 dataset | Travis + Matthew | Build the 3 number cards. Add results/class_distribution.png. |
| 4 models | Alan + Tanner | Build the 3 model cards. Shrink the bullets into speaker notes. |
| 5 results | Alan | Add the 2 charts + accuracy line once training is done. |
| 6 future | Tanner | Replace the paragraph with the 4 short phrases. |
| 7 demo | Matthew | Add the app screenshot. Practice the live flow as backup. |
| 8 conclusion | Erwin | Add the 3 closing lines. |
| 9 | anyone | Delete, or make it a "thank you" in slide 1's style. |

## Making it look like slide 1

- Headline banners: copy slide 1's brown rectangle + white lowercase text and paste
  it onto each slide, then edit the words. Copying the actual element keeps the
  font, size, and color identical.
- White cards: copy the white "Scan smarter. Skin clearer." box from slide 1 and
  resize it. Same reason.
- Decorative elements: in Canva Elements, search "zen stones stack", "cream smear",
  "green branch". Pick ones that match slide 1's soft photographic style. One or two
  per slide at most, tucked into a corner, never behind text.
- Charts: drag the png in, then put a white rectangle behind it (slide 1's box style)
  so it reads as a framed card.

## Words rule

If a slide ends up with more than about 15 words, cut it. The test: someone in the
back row should be able to read the whole slide in 3 seconds. Everything else is
said out loud, and DECK.md already wrote those speaker notes for you. Paste them
into Canva via Notes at the bottom of each page.

## Charts produced by the notebook

| File in results/ | Use on |
|---|---|
| class_distribution.png | slide 3 |
| skin_type_confusion_matrix.png | slide 5 |
| yolo_sample_detections.png | slide 5 |
| acne_type_confusion_matrix.png | backup, if asked about acne model |
| skin_type_training_curves.png, acne_type_training_curves.png | backup, if asked about training |
| grad_cam_skin_type.png | great for a "how do we know it works" question |
