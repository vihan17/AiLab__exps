import heapq
import re
from typing import List, Tuple, Dict, Optional
import math

class PlagiarismDetector:
    def __init__(self):
        self.skip_penalty = 10  # Penalty for skipping a sentence

    def preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize and normalize text into sentences.
        """
        # Split into sentences (simple approach - can be enhanced with NLTK)
        sentences = re.split(r'[.!?]+', text)

        # Normalize each sentence
        normalized_sentences = []
        for sentence in sentences:
            # Convert to lowercase and remove extra whitespace
            sentence = sentence.lower().strip()
            # Remove punctuation (keep only alphanumeric and spaces)
            sentence = re.sub(r'[^\w\s]', '', sentence)
            if sentence:  # Only add non-empty sentences
                normalized_sentences.append(sentence)

        return normalized_sentences

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Compute the Levenshtein distance between two strings.
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def sentence_similarity(self, s1: str, s2: str) -> float:
        """
        Compute similarity between two sentences (0 to 1 scale).
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        distance = self.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)

    def heuristic(self, doc1: List[str], doc2: List[str], i: int, j: int) -> float:
        """
        Heuristic function for A* search - estimates remaining alignment cost.
        Uses minimum possible edit distance for remaining sentences.
        """
        remaining_doc1 = len(doc1) - i
        remaining_doc2 = len(doc2) - j

        # The heuristic is the minimum number of operations needed
        # which is the absolute difference in remaining sentences
        min_remaining = min(remaining_doc1, remaining_doc2)
        max_remaining = max(remaining_doc1, remaining_doc2)

        # Minimum cost: align min_remaining sentences perfectly + skip the rest
        return (max_remaining - min_remaining) * self.skip_penalty

    def a_star_alignment(self, doc1: List[str], doc2: List[str]) -> List[Tuple[Optional[int], Optional[int], float]]:
        """
        Perform A* search to find optimal alignment between two documents.
        Returns list of alignments: (doc1_index, doc2_index, similarity_score)
        """
        # Priority queue: (f_score, g_score, i, j, path)
        open_set = []

        # g_score: cost from start to current state
        g_scores = {(0, 0): 0}

        # f_score = g_score + heuristic
        f_score = 0 + self.heuristic(doc1, doc2, 0, 0)
        heapq.heappush(open_set, (f_score, 0, 0, 0, []))  # f, g, i, j, path

        best_path = None
        best_cost = float('inf')

        while open_set:
            current_f, current_g, i, j, path = heapq.heappop(open_set)

            # Check if we reached the goal state
            if i == len(doc1) and j == len(doc2):
                if current_g < best_cost:
                    best_cost = current_g
                    best_path = path
                continue

            # If we already found a better path to this state, skip
            if current_g > g_scores.get((i, j), float('inf')):
                continue

            # Generate possible transitions
            transitions = []

            # Option 1: Align current sentences (if both have sentences left)
            if i < len(doc1) and j < len(doc2):
                similarity = self.sentence_similarity(doc1[i], doc2[j])
                cost = 1 - similarity  # Convert similarity to cost
                new_g = current_g + cost
                transitions.append((i + 1, j + 1, new_g, path + [(i, j, similarity)]))

            # Option 2: Skip sentence in doc1
            if i < len(doc1):
                new_g = current_g + self.skip_penalty
                transitions.append((i + 1, j, new_g, path + [(i, None, 0)]))

            # Option 3: Skip sentence in doc2
            if j < len(doc2):
                new_g = current_g + self.skip_penalty
                transitions.append((i, j + 1, new_g, path + [(None, j, 0)]))

            # Evaluate transitions
            for new_i, new_j, new_g, new_path in transitions:
                if new_g < g_scores.get((new_i, new_j), float('inf')):
                    g_scores[(new_i, new_j)] = new_g
                    f_score = new_g + self.heuristic(doc1, doc2, new_i, new_j)
                    heapq.heappush(open_set, (f_score, new_g, new_i, new_j, new_path))

        return best_path if best_path else []

    def detect_plagiarism(self, text1: str, text2: str, similarity_threshold: float = 0.8) -> Dict:
        """
        Main function to detect plagiarism between two texts.
        """
        # Preprocess texts
        doc1 = self.preprocess_text(text1)
        doc2 = self.preprocess_text(text2)

        print(f"Document 1: {len(doc1)} sentences")
        print(f"Document 2: {len(doc2)} sentences")

        # Perform A* alignment
        alignment = self.a_star_alignment(doc1, doc2)

        # Analyze results for plagiarism
        plagiarism_pairs = []
        total_similarity = 0
        aligned_pairs = 0

        for doc1_idx, doc2_idx, similarity in alignment:
            if doc1_idx is not None and doc2_idx is not None:
                aligned_pairs += 1
                total_similarity += similarity

                if similarity >= similarity_threshold:
                    plagiarism_pairs.append({
                        'doc1_sentence': doc1[doc1_idx] if doc1_idx < len(doc1) else "OUT_OF_BOUNDS",
                        'doc2_sentence': doc2[doc2_idx] if doc2_idx < len(doc2) else "OUT_OF_BOUNDS",
                        'similarity': similarity,
                        'doc1_index': doc1_idx,
                        'doc2_index': doc2_idx
                    })

        avg_similarity = total_similarity / aligned_pairs if aligned_pairs > 0 else 0

        return {
            'plagiarism_pairs': plagiarism_pairs,
            'total_aligned_pairs': aligned_pairs,
            'average_similarity': avg_similarity,
            'plagiarism_percentage': (len(plagiarism_pairs) / aligned_pairs * 100) if aligned_pairs > 0 else 0,
            'alignment_path': alignment
        }

def run_test_cases():
    detector = PlagiarismDetector()

    print("=" * 60)
    print("TEST CASE 1: Identical Documents")
    print("=" * 60)

    text1 = "This is the first sentence. This is the second sentence. This is the third sentence."
    text2 = "This is the first sentence. This is the second sentence. This is the third sentence."

    result = detector.detect_plagiarism(text1, text2)
    print(f"Plagiarism pairs found: {len(result['plagiarism_pairs'])}")
    print(f"Average similarity: {result['average_similarity']:.3f}")
    print(f"Plagiarism percentage: {result['plagiarism_percentage']:.1f}%")

    print("\n" + "=" * 60)
    print("TEST CASE 2: Slightly Modified Document")
    print("=" * 60)

    text1 = "The quick brown fox jumps over the lazy dog. Programming is fun and challenging."
    text2 = "A quick brown fox leaps over the lazy dog. Programming is enjoyable and challenging."

    result = detector.detect_plagiarism(text1, text2)
    print(f"Plagiarism pairs found: {len(result['plagiarism_pairs'])}")
    print(f"Average similarity: {result['average_similarity']:.3f}")
    print(f"Plagiarism percentage: {result['plagiarism_percentage']:.1f}%")

    # Print plagiarism pairs
    for i, pair in enumerate(result['plagiarism_pairs']):
        print(f"\nPlagiarism Pair {i + 1}:")
        print(f"  Doc1: {pair['doc1_sentence']}")
        print(f"  Doc2: {pair['doc2_sentence']}")
        print(f"  Similarity: {pair['similarity']:.3f}")

    print("\n" + "=" * 60)
    print("TEST CASE 3: Completely Different Documents")
    print("=" * 60)

    text1 = "Machine learning is a subset of artificial intelligence. Python is a popular programming language."
    text2 = "The weather today is sunny and warm. Cooking requires patience and practice."

    result = detector.detect_plagiarism(text1, text2)
    print(f"Plagiarism pairs found: {len(result['plagiarism_pairs'])}")
    print(f"Average similarity: {result['average_similarity']:.3f}")
    print(f"Plagiarism percentage: {result['plagiarism_percentage']:.1f}%")

    print("\n" + "=" * 60)
    print("TEST CASE 4: Partial Overlap")
    print("=" * 60)

    text1 = "Data structures are important for efficient programming. Algorithms help solve complex problems. Python is versatile."
    text2 = "Computer science involves many concepts. Algorithms help solve complex problems. Python is versatile and powerful."

    result = detector.detect_plagiarism(text1, text2)
    print(f"Plagiarism pairs found: {len(result['plagiarism_pairs'])}")
    print(f"Average similarity: {result['average_similarity']:.3f}")
    print(f"Plagiarism percentage: {result['plagiarism_percentage']:.1f}%")

    # Print plagiarism pairs
    for i, pair in enumerate(result['plagiarism_pairs']):
        print(f"\nPlagiarism Pair {i + 1}:")
        print(f"  Doc1: {pair['doc1_sentence']}")
        print(f"  Doc2: {pair['doc2_sentence']}")
        print(f"  Similarity: {pair['similarity']:.3f}")
