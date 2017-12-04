![Alt text](poster.png?raw=true "Title")

Paper link: http://papers.nips.cc/paper/6919-language-modeling-with-recurrent-highway-hypernetworks.pdf

NOTE: Use pytorch for 1.12_2 for now. Has not been confirmed working with 2.0. If you're just interested in the clean RHN cell implementation, that should work, potentially with small API updates. I do plan on updating, but not until after the conference.

12/3/17: Final correctness checks. Had to drop the batch size to 200 due to memory constraints. Our model obtains 1.195 BPC after 650 epochs. The RHN takes 1020 epochs to converge and likely still has not converged--that said, the perplexity is constant to the third digit since epoch 650, so this is not a concern. Run the respective MainHRHN and MainRHN files to reproduce. Also updated poster (old version still in the repo) and fixed the inconsistent date labeling below.

11/28/17: Major code changes and correctness checks--final tests are running now and look good so far. This should be the second to last update. Added MainRHN.py and MainHRHN.py for easier running of separate models. 

11/16/17: Added poster. This is a very abbreviated version of the full paper largely intended for a live audience, but it still serves as a fairly good bare-bones math-only representation of the project. The tests will take a few more days--the good new is, the code will be very clean and very short :)

11/8/17: Initial commit. Code cleaned from original repo. I still have some polishing to do on the utils/nlp libraries and train/test system. I have not yet tested consistency with the original repo; this will begin tomorrow, but runs take 2-3 days. Feel free to take a look at the core modules until then, but I don't advise cloning yet.

11/4/17: Stub repo for associated NIPS Paper. Code is currently being cleaned and tested for upload--give it 4 days to a week
