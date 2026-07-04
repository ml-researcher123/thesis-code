# The Compounding Bottleneck — explained simply

This is a plain-language walkthrough of the paper in `paper/latex/iclr2026/main.tex`. No machine
learning background assumed — every term is defined before it's used.

## The setup: how "efficient RAG" actually works

A lot of AI chat systems (like a chatbot that can answer questions about your company's
documents) use a technique called **Retrieval-Augmented Generation**, or **RAG**. Instead of
having the AI memorize everything, RAG looks things up on the fly, similar to how you'd search a
library before answering a question rather than trying to remember every book by heart. RAG has
two steps:

1. **Retrieval** — given a question, find the handful of documents (out of possibly millions)
   that are actually relevant. This is done by turning both the question and every document into
   a list of numbers (a "vector" or "embedding") and finding documents whose numbers are
   mathematically close to the question's numbers. Think of it like giving every document and
   every question a GPS coordinate in some abstract space, and retrieval just finds the nearest
   documents to the question's coordinate.

2. **Compression** — once you've found the relevant documents, they're often too long to feed
   directly to the AI (it's slow and expensive to process a lot of text). So a lot of "efficient"
   RAG systems squeeze each document down into a small number of these same numeric vectors — a
   kind of lossy summary in number form — before handing it to the AI that writes the final
   answer.

The key thing both of these steps have in common: **they both squeeze information through a
fixed-size list of numbers.** Retrieval squeezes a whole document into a vector of a fixed length,
say 1024 numbers. Compression squeezes a document into an even smaller vector (or a handful of
small vectors) before generation. Both are what the paper calls **bottlenecks** — a narrow point
that everything has to pass through, and if the bottleneck is too narrow, information gets lost no
matter how good the rest of the system is.

## The core question this paper asks

Nobody had really asked: what happens when you put *two* of these narrow bottlenecks back to
back? Does the damage from bottleneck #1 (retrieval) and bottleneck #2 (compression) just add up
in a predictable way, or does something worse happen when they're chained together? And if two
bottlenecks share a fixed "budget" (say, you can afford 128 numbers total, split however you
like between the two), what's the best way to split it? That's the whole paper. It has four
findings.

---

## Finding 1: Both bottlenecks have a hard capacity limit (a "wall")

Imagine you're trying to describe every person in a room using nothing but a coordinate on a map
— nowhere on the map represents "everyone who likes jazz AND everyone who owns a dog," you'd need
enough distinct regions to capture every possible combination of traits. If the map is too small
(too few numbers, i.e. too low-dimensional), there are physically groups of people you cannot
single out no matter how cleverly you place points on it.

The same limitation applies to retrieval and compression. If you use too few numbers, there are
*mathematically guaranteed* patterns of "these documents belong together" that a retrieval system
can never correctly separate, no matter how well-trained it is. This isn't a training problem, it's
a geometry problem — more numbers (higher-dimensional vectors) means you can represent more
complex patterns, and below some "critical size," some patterns are simply impossible to
represent. The paper calls this critical size the **wall**.

The surprising part is how well this predicts real behavior. We:

- Verified this mathematically-guaranteed wall exists in a best-case, idealized setting (using
  vectors that are free to be anything, not just what a real AI model would produce).
- Then showed it also shows up in **real, off-the-shelf embedding models** — recall (how many of
  the right documents get found) climbs steeply as the vector size grows, then flattens out. For
  one popular embedding model (`mxbai-embed-large`), performance basically stopped improving past
  512 numbers even though the model uses 1024 — meaning **half of what it computes is wasted**.
- Checked this held up across a second embedding model, a second topic domain, and even under the
  mathematically "best possible" way of shrinking a vector (not just the common shortcut method)
  — so it's not a fluke of one model or one trick.
- And most convincingly, tested it on a benchmark called **LIMIT**, which was specifically built
  by another research group to be an adversarial worst-case test for this exact limitation.
  Three different real embedding models, when asked to find 100 candidate documents out of 50,000,
  only found the right one **2.8% to 8.2%** of the time. That's a near-total failure, and it
  happened for all three models — which is strong evidence this is a fundamental limit of the
  whole approach, not a bug in one particular model.
- The compression step has the exact same kind of wall: cram too many distinct facts into too few
  numbers and past a certain point, the system simply can't tell them apart anymore. A single
  vector holding just one document's worth of information could recall a single fact correctly 98%
  of the time — but recall dropped to 20% (barely better than guessing) once it had to hold 16
  facts.

## Finding 2: Chaining two imperfect stages together is worse than either stage alone (they "compound")

Here's the part that gives the paper its name. Suppose your retrieval step gets the right answer
90% of the time by itself, and your compression step (given the right document) also gets it right
90% of the time by itself. You might assume the whole pipeline gets it right somewhere close to
90% of the time too. It doesn't — if the two steps' successes and failures are independent of each
other, the *combined* system only succeeds about 90% × 90% = 81% of the time, because both steps
have to succeed for the final answer to be right.

That already sounds bad, but the paper shows it's often worse in practice, because if a document
happens to be hard for both retrieval *and* compression at the same time (rather than randomly
independent), the pipeline's success rate can drop below even that 81% estimate. Whether that
happens depends on a specific number the paper measures directly: **the correlation between
"which documents are hard to retrieve" and "which documents are hard to compress."** If this
correlation is close to zero, the "just multiply the two success rates" estimate is accurate. If
retrieval-hard documents also tend to be compression-hard documents, things get much worse than
that. If it's the opposite — retrieval-hard documents are actually compression-easy, and vice
versa — the combined system does a bit better than the simple multiplication would suggest.

Measuring this on a real pipeline (a real retrieval model plus a real compression model working
together), the paper finds this correlation is **essentially zero** — meaning real systems, at
least in the setup tested, land almost exactly on the "just multiply the two success rates"
prediction, to within about a tenth of a percentage point. Concretely, with a modest number budget
(128 numbers total shared between retrieval and compression), the two stages individually succeed
63% and 89% of the time — but the full pipeline only succeeds 23.4% of the time. That's the
"compounding": the finished system is dramatically weaker than either half of it looks on its own,
and this gap shrinks the more numbers (budget) you give the whole system.

## Finding 3: Given a fixed budget, there's a "correct" way to split it — and most real systems get it wrong

If you have a fixed total budget of numbers to spend, and it has to be divided between the
retrieval vector and the compression vector, how should you split it? The paper's answer, backed
by the "wall" from Finding 1: spend just enough on retrieval to clear its wall (get past the point
where more numbers stop helping), and put everything left over into compression.

Most deployed real-world systems do roughly the opposite of this — they use a very large retrieval
vector (768 to 1024 numbers) and a tiny compression code (as few as 1 to 8 numbers). Given
Finding 1, that means these systems are pouring a large chunk of their budget into retrieval
dimensions that have already stopped helping, while starving the compression stage of the numbers
it actually needs. The paper shows that re-allocating the exact same total budget — just moving
numbers from an over-provisioned retrieval vector to an under-provisioned compression code — can
noticeably improve overall accuracy at no extra computational cost.

## Finding 4: A tempting fix works in theory but not on today's strong models (an honest negative result)

Given that a single vector runs into a hard wall, an obvious idea is: what if, instead of one
vector, you used several specialized ones? For instance, one vector-space ("lens") specialized for
matching by profession, another for matching by location, another for matching by hobby, and so
on — and a lightweight "router" that figures out which lens is relevant to a given question and
uses just that one.

In an idealized, worst-case test, this idea works really well: it broke through the single-vector
wall using half the total budget of a single, unspecialized vector. But — and this is important —
the paper also tested this idea on a strong, real, pretrained embedding model, and found that the
advantage **disappears**. A single, ordinary vector (just trained a bit more cleverly) did just as
well as the multi-lens approach, and actually did *better* when the number budget was tight. A
generic "multiple unspecialized vectors" version (without the clever routing) was clearly worse
than either.

The likely reason: a strong pretrained model has already done a lot of the hard work of untangling
different attributes internally, so a single well-trained vector can still pull them apart without
needing separate specialized lenses. The multi-lens trick mattered in an artificial worst case, but
it isn't a reliable practical fix for today's real systems. The paper reports this as an honest
negative finding rather than hiding it, since it sharpens the more actionable message from Finding
3: if you want a better RAG system today, the reliable lever is **how you split your numeric
budget between retrieval and compression**, not a fancier vector representation.

## the follow-up i had to run: does any of this hold when a real model answers real questions?

so once i had those four findings i couldn't quite leave it there, because if i'm being honest most
of what i'd shown up to this point lived in fairly controlled setups — the walls came out of "free"
vectors that are allowed to be literally anything, the compounding and the budget-split were
measured on a little corpus i built myself, and the stand-in for the reader in the compression
tests was a lightweight probe rather than an actual language model writing out answers, and that's
genuinely great for isolating the mechanism cleanly, but a skeptical reader (or just a more
suspicious version of me) would immediately push back with "sure, but does any of this survive when
a real system answers real questions and you score it the way people actually score QA?", and i
didn't really have a clean answer to that yet, so i went and built one. what i did was take real
multi-hop question-answering benchmarks — HotpotQA first, and then 2WikiMultihopQA as a second one
just so i wasn't fooling myself on a single dataset — and run an actual pipeline end to end: a real
retriever pulling passages out of a shared pool, a real compression step that has to decide which
sentences survive into a tight reading budget, and then a real off-the-shelf reader model
(Qwen2.5-Instruct) that genuinely generates the answer, which i score against the gold answer with
the standard exact-match and F1 metrics, so there's nothing artificial left in the loop anymore,
it's basically the same shape as a real efficient-RAG system.

and the nice thing (nice for the theory, maybe less nice if you're the one running these systems)
is that the compounding shows up almost exactly like the synthetic story said it would — at a tight
budget the best possible split of the numbers between retrieval and compression still loses somewhere
around 0.05 in F1 compared to what either stage looks like on its own, and that gap shrinks as you
loosen the budget, which is the same shape i'd seen on the made-up corpus, so it wasn't an artifact
of my toy setup after all. the budget-split story holds too, and this is the part i actually find
kind of satisfying: if you keep pouring more and more of the budget into the retrieval side, the
answer quality climbs for a while and then it genuinely starts coming back down, because past some
point you're just starving the compression stage and the answer stops making it into the reader's
context at all — so the "the best split is somewhere in the middle, not at either extreme" claim is
something you can literally watch happen to a real reader's accuracy, it peaks and then falls off as
you over-invest in one stage. and it wasn't a fluke of one benchmark either, since the second
dataset gave me essentially the same number (that ~0.05 gap again), which is about as much
cross-dataset agreement as i could reasonably hope for.

the last worry i wanted to kill off was scale, because someone will always say "yeah but a bigger,
smarter reader will just paper over whatever evidence got dropped, so who really cares", and it's a
fair thing to say, so i ran the whole thing again at three reader sizes — 1.5B, then 3B, then 7B
parameters, which is roughly a 5x range — and the compounding gap just sort of stayed put, right
around that same 0.05, it didn't quietly shrink away as the reader got stronger and it didn't flip
sign. and honestly that's the cleanest way i can say what kind of problem this even is: it's an
information problem, not a reasoning problem, because if the evidence got squeezed out at the two
bottlenecks then it's simply not in front of the reader anymore, and no amount of the reader being
clever is going to bring back something that isn't there (this is actually where it quietly parts
ways with some related work on trimming context down, where a stronger reader tends to wash the
benefit out — here it goes the other way, if anything a stronger reader makes the missing-evidence
penalty a touch *more* visible, since it's good enough that it would have used that evidence if it
had ever reached it). so between the two datasets and the three reader sizes i'm now fairly
comfortable saying the compounding and the middle-of-the-road budget split aren't quirks of a
synthetic benchmark, they're what actually happens when a real model tries to answer real questions
under a real budget.

---

## Why this matters, in one paragraph

If you build or use a RAG-based AI system that's optimized to be fast and cheap (which almost
every production system is, because processing full documents for every question is
prohibitively slow and expensive), this paper says: (1) there's a hard, unavoidable capacity limit
in both the "find relevant documents" step and the "shrink them down" step, and you can locate that
limit cheaply just by testing at a few different vector sizes; (2) chaining those two limited
steps together makes the overall system meaningfully weaker than either step looks in isolation,
by an amount you can actually predict — and this isn't just a synthetic-benchmark quirk, it holds
when a real reader model answers real multi-hop questions scored by exact-match/F1, across reader
sizes from 1.5B to 7B, precisely because it's an information loss the reader can't reason its way
back out of; (3) most real systems split their numeric budget
badly — favoring an oversized retrieval step over a starved compression step — and fixing that
split is free, since it doesn't cost extra compute, just a smarter allocation of the same budget;
and (4) fancier multi-part representations are not a shortcut around this — they help in
adversarial worst cases but not, so far, on the strong models people actually use, so the practical
job right now is getting the numeric budget allocation right.
