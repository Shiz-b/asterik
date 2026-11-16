# COMP9444 Project 014 — Presentation Speech Script (Sudam)

> **Presenter:** Sudam Attanayake (`z5363513`)  
> **Team:** Asterik and Obelix  
> **Slides Covered:** Slide 1 (Intro), Slide 2 (Motivation), Slide 3 (Problem Statement), Slide 16 (Conclusion)  
> **Deck:** [`COMP9444_Presentation_Asterik_and_Obelix.pptx`](COMP9444_Presentation_Asterik_and_Obelix.pptx)  
> **Estimated Total Speaking Time:** ~3 minutes  

---

## Marking Criteria Checklist (Presentation: 8 Marks Total)
- **Individual (3 marks):** Contribution to the presentation (`1.5`) + Clearly spoken and understandable (`1.5`)
- **Group (5 marks):** Clear problem statement (`1.0`), Data/EDA (`1.0`), Models/Methods (`1.0`), Results/Error analysis (`1.0`), Slide quality (`1.0`)
- **Tutor guidance:** Keep slides clean and minimal; deliver the explanations, reasoning, and transitions verbally.

---

## 1. Slide 1 — Intro *(~15 seconds)*

Hi everyone, we're **Asterik and Obelix**—that's Shabs, Atul, Shetal, and myself, Sudam. Today we're presenting **Project 014: Automated Marine Debris Detection using Deep Learning**, where we investigate how accurately and quickly modern object detectors can locate underwater trash in real deep-sea footage.

---

## 2. Slide 2 — Motivation *(~45 seconds)*

Why does this problem matter? Every year, growing volumes of marine litter settle on the seafloor, entangling marine animals, leaching microplastics, and disrupting fragile ecosystems.

Today, surveying and clearing that debris still depends heavily on human divers or remotely operated vehicles—ROVs—piloted manually. As you can imagine, sending divers or operators down across vast ocean areas is slow, expensive, and inconsistent from operator to operator.

If we equip autonomous underwater vehicles with deep-learning vision, they can scan large areas continuously. **And as you can see in this real JAMSTEC frame on the right**, a piece of plastic waste sits right beside sea shells on the seabed—so an automated detector can't just flag any foreground shape; it has to tell **man-made debris apart from marine life** so clean-up robots don't damage coral or fauna.

---

## 3. Slide 3 — Problem Statement *(~65–75 seconds)*

That brings us to our exact problem statement. Given a single underwater RGB frame, our task is to **detect, classify, and localise** every object with a bounding box across **16 classes** in the `TrashCan-Material` dataset—covering specific debris materials like plastic and metal, alongside animals, plants, and the ROV itself.

Looking at prior research, **three specific gaps** shaped our project goals:

- **First, sensing and label granularity:** Earlier robotic studies used forward-looking sonar or terrestrial LiDAR. Sonar works in zero visibility, but it loses colour and fine surface texture—so it can't tell plastic apart from metal—while LiDAR is absorbed by water and fails on glass. Early RGB benchmarks like Fulton et al. solved this with cameras, but grouped everything into just three coarse classes. **Our goal** is fine-grained, **16-class optical RGB material detection**.
- **Second, evaluation rigour:** Foundational TrashCan work by Hong et al. and several recent YOLO studies split images **randomly at the frame level**—meaning nearly identical frames from the same video clip appear in both training and validation—and recent papers often report only `mAP@0.5` without runtime speed. **Our goal** is a **strict video-disjoint benchmark** across all 312 videos—with zero clip overlap—evaluated on full **COCO AP metrics and FPS**.
- **Third, underwater degradation and class skew:** Deep-sea footage suffers from heavy blue-green colour cast, murky low contrast, and an extreme **297-to-1** training imbalance—and Đuraš et al. showed that naive post-hoc image enhancement actually hurts detection accuracy. **So our goal** is to tackle both directly inside training using **physics-aware underwater augmentations and Repeat-Factor Sampling**.

*(Point briefly to the bottom Input → Output visual)* Here's what that looks like end-to-end: a murky ROV frame goes in, and the model outputs class labels and tight boxes separating `trash: plastic` from the `eel` and the `rov` itself. I'll now hand over to my teammate to walk through the literature review in detail.

---

## 4. Slide 16 — Conclusion *(~55–65 seconds — after Discussion)*

To wrap everything up, let's tie our findings directly back to the three goals we set in our problem statement.

- **On 16-class real-time detection:** We benchmarked four detectors end-to-end. Our improved two-stage Faster R-CNN v2 reaches **21.4% Test AP50 at 49 frames per second**—well above real-time video rates—while our lightweight MobileNetV3 baseline offers a **127-FPS** option for compute-constrained edge ROVs.
- **On rigorous, leak-free evaluation:** Across **312 strictly video-disjoint ROV clips**, two-stage Faster R-CNN outperformed single-stage RetinaNet by **9.8 AP50 points** (**21.4% versus 11.6%**). Because underwater frames are sparse—averaging only **1.7 objects per image**—two-stage region proposal filtering handles cluttered seafloor backgrounds much more reliably than dense single-stage anchor grids.
- **And on underwater visibility and class imbalance:** Pairing **Repeat-Factor Sampling** with **in-training colour and contrast augmentation** lifted Test AP50 by **2.4 points**, boosted rare classes like `shells` by **18.5 points** and `eel` by **4.5 points**, and reduced missed detections in dark frames from **65.3% down to 63.3%**.

**The bottom line:** Combining two-stage proposal filtering, Repeat-Factor Sampling, and in-training underwater augmentation gives us a fast, accurate, and genuinely leak-free baseline for automated marine debris monitoring.

*(Advance to Slide 17 — Questions)* Thank you for listening—we're happy to take any questions.

---

## Quick Delivery Tips
- Use the bold slide figures (`16 classes`, `312 videos`, `297-to-1`, `21.4% @ 49 FPS`, `+9.8 AP50`, `shells +18.5%`) as visual cues rather than reading the table verbatim.
- Pause briefly between each of the three points on Slide 3 and Slide 16 so the symmetry between **Problem Statement** and **Conclusion** lands clearly for the marker.
