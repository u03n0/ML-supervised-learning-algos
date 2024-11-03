# K Nearest Neighbor

## High Level
The K Nearest Neighbor is a simple supervised algorithm that checks the closeness of 'k' amount of plotted neighbors (vectors) to a current point (vector).

If we had 300 plotted vectors, each representing a document (spam or normal email) and we gave it a new vector to plot (unseen email), based on its location on 
a 2-D graph with X and Y coordinates, how could we determine it belongs to a particular cluster (grouping) ?

Well, if we were to measure the distance between this new unseen point and all others, we can then take the top k amount who have the shortest distance (euclidean distance) to our unseen point,
or those that have the highest similarity (cosine similarity). Those that are closer together, are grouped together, probably share a lot in common.

## Terminology
- Cosine-Similarity: is the dot product of two vectors, divided by the product of their magnitudes.
` A * B / ||A|| * ||B||`
- corpus : a corpus is a collection of documents
- document: is a single unit of text (str)
- term: is a single element (word) in a document

- term-frequency (TF): the TF is calculated on a document level. It is the occurence of a term (t) given a document (d) divided by the total count of terms in the document. tf(t,d) = f<sub>t,d</sub>
  
