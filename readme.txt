### **Understanding How Histogram Correlation Ensures Image Appending**  
When checking if two images (i.e., chessboard squares) are **part of the same piece**, we need to determine whether their edges **align properly**.  

Using histogram correlation on **edge strips** helps in this process. Let’s break it down:

---

## **1️⃣ What Does "Appending" Mean?**
For two images (e.g., two adjacent chessboard squares) to be considered **appending**:
- Their edges must match up **seamlessly**.  
- The **left side** of one piece should continue smoothly into the **right side** of the adjacent piece.  
- Similarly, the **bottom edge** of one piece should connect to the **top edge** of the piece below it.

If two images are part of **different pieces**, their edges should **look different** (different pixel intensity patterns).

---

## **2️⃣ How Does Histogram Correlation Work?**
A **histogram** represents the distribution of pixel intensities in an image.  
- For grayscale edge images, it shows how many pixels exist at each brightness level (0-255).  
- If two images are part of the same piece, their edge histograms will be **very similar**.  
- If they are separate, their histograms will look **different**.  

### **Steps in Edge Histogram Correlation:**
1. **Extract a 5-pixel-wide strip from both images**  
   - If checking **left**, compare:  
     - **Rightmost 5 pixels** of the left image  
     - **Leftmost 5 pixels** of the right image  
   - If checking **top**, compare:  
     - **Bottom 5 pixels** of the top image  
     - **Top 5 pixels** of the bottom image  

2. **Convert the strips to grayscale** (if they aren’t already)  

3. **Compute histograms for both strips**  
   - A histogram shows how many pixels exist at each intensity level (0-255).  
   - Example:  
     - If one strip has many **bright edges** and the other has **dark edges**, their histograms will look different.  

4. **Compare histograms using correlation**  
   - **If correlation > 0.7**, the two strips are likely part of the same piece (their edges align).  
   - **If correlation < 0.7**, the strips are different, meaning the pieces are separate.  

---

## **3️⃣ Why Does This Ensure Proper Appending?**
### **✅ Case 1: Images Are Appending (Same Chess Piece)**
**Example:** A rook covering two squares.  
- The edges of the rook should continue **smoothly** from one square to the next.  
- The 5-pixel-wide strips from both images will have **similar edge intensities**.  
- Their histograms will be **highly correlated** (close to 1.0).  

**Result:** We correctly determine that these two squares belong to the same piece.

---

### **❌ Case 2: Images Are Not Appending (Different Pieces)**
**Example:** A black queen next to an empty square.  
- The edges of the queen will be **very different** from the edges of an empty square.  
- The 5-pixel-wide strips will have **different edge intensities**.  
- Their histograms will be **uncorrelated** (close to 0.0).  

**Result:** We correctly determine that these two squares do NOT belong to the same piece.

---

## **4️⃣ Why Not Use Structural Similarity Index (SSIM)?**
- **SSIM compares whole images, not just edges.**  
- It’s sensitive to minor variations and lighting changes.  
- It falsely groups non-matching pieces together.  

### **Histogram correlation is better because:**
✅ It only focuses on edge patterns.  
✅ It’s **rotation-invariant** (small shifts don’t affect correlation).  
✅ It’s **less sensitive to minor lighting changes**.  

---

## **🔍 Conclusion: Ensuring Proper Appending**
Using **edge histograms**:
- If two adjacent squares have similar edges → They are **part of the same piece**.  
- If their edges are different → They **belong to separate pieces**.  

This method **accurately detects if a chess piece spans multiple squares**, ensuring **correct classification**. 🚀♟️