from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import product, combinations
from statistics import mean
from typing import List, Iterable, Set, Generator

from nltk.corpus import wordnet

from grimjack.model import RankedDocument, Query
from grimjack.modules import RerankingContext


def strictly_greater(x, y):
    if x > y:
        return 1
    elif y > x:
        return -1
    return 0


def strictly_less(x, y):
    if y > x:
        return 1
    elif x > y:
        return -1
    return 0


def approximately_equal(*args, margin_fraction: float = 0.1):
    """
    True if all numeric args are
    within (100 * margin_fraction)% of the largest.
    """

    abs_max = max(args, key=lambda item: abs(item))
    if abs_max == 0:
        # All values must be 0.
        return True

    b = [abs_max * (1 + margin_fraction), abs_max * (1 - margin_fraction)]
    b_min = min(b)
    b_max = max(b)
    return all(b_min < item < b_max for item in args)


def all_query_terms_in_documents(
        context: RerankingContext,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument
):
    query_terms = context.term_set(query.title)
    document1_terms = context.term_set(document1.content)
    document2_terms = context.term_set(document2.content)

    if len(query_terms) <= 1:
        return False

    return (
            len(query_terms & document1_terms) == len(query_terms) and
            len(query_terms & document2_terms) == len(query_terms)
    )


def same_query_term_subset(
        context: RerankingContext,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument
) -> bool:
    """
    Both documents contain the same set of query terms.
    """

    query_terms = context.term_set(query.title)
    document1_terms = context.term_set(document1.content)
    document2_terms = context.term_set(document2.content)

    if len(query_terms) <= 1:
        return False

    in_document1 = query_terms & document1_terms
    in_document2 = query_terms & document2_terms

    # Both contain the same subset of at least two terms.
    return (in_document1 == in_document2) and len(in_document1) > 1


def approximately_same_length(
        context: RerankingContext,
        document1: RankedDocument,
        document2: RankedDocument,
        margin_fraction: float = 0.1
) -> bool:
    return approximately_equal(
        len(context.terms(document1.content)),
        len(context.terms(document2.content)),
        margin_fraction
    )


def synonym_set_similarity(
        term1: str,
        term2: str,
        smoothing: int = 0
) -> float:
    cutoff = smoothing + 1
    synonyms_term1 = wordnet.synsets(term1)[:cutoff]
    synonyms_term2 = wordnet.synsets(term2)[:cutoff]

    n = 0
    similarity_sum = 0

    for synonym1, synonym2 in product(synonyms_term1, synonyms_term2):
        similarity = wordnet.wup_similarity(synonym1, synonym2)
        if similarity is not None:
            similarity_sum += similarity
            n += 1

    if n == 0:
        return 0

    return similarity_sum / n


def vocabulary_overlap(vocabulary1: Set[str], vocabulary2: Set[str]):
    """
    Vocabulary overlap as calculated by the Jaccard coefficient.
    """
    intersection_length = len(vocabulary1 & vocabulary2)
    if intersection_length == 0:
        return 0
    return (
            intersection_length /
            (len(vocabulary1) + len(vocabulary2) - intersection_length)
    )


def average_between_query_terms(
        query_terms: Set[str],
        document_terms: List[str]
) -> float:
    query_term_pairs = set(combinations(query_terms, 2))
    if len(query_term_pairs) == 0:
        # Single-term query.
        return 0

    number_words = 0
    for item in query_term_pairs:
        element1_position = document_terms.index(item[0])
        element2_position = document_terms.index(item[1])
        number_words += abs(element1_position - element2_position - 1)
    return number_words / len(query_term_pairs)


def take_closest(l: List[int], n: int):
    """
    Return closest value to n.
    If two numbers are equally close, return the smallest number.

    It is assumed that l is sorted.
    See: https://stackoverflow.com/questions/12141150
    """
    position = bisect_left(l, n)
    if position == 0:
        return l[0]
    if position == len(l):
        return l[-1]
    before = l[position - 1]
    after = l[position]
    if after - n < n - before:
        return after
    else:
        return before


def query_term_index_groups(
        query_terms: Set[str],
        document_terms: List[str]
) -> Generator[List[int]]:
    indexes = defaultdict(list)
    for index, term in enumerate(document_terms):
        if term in query_terms:
            indexes[term].append(index)
    for term in query_terms:
        other_query_terms = query_terms - {term}
        for index in indexes[term]:
            group = [index] + [
                take_closest(indexes[other_term], index)
                for other_term in other_query_terms
                if len(indexes[other_term]) > 0
            ]
            yield group


def closest_grouping_size_and_count(
        query_terms: Set[str],
        document_terms: List[str]
):
    index_groups = query_term_index_groups(query_terms, document_terms)

    # Number of non-query terms within groups.
    non_query_term_occurrences = [
        len([
            term
            for term in document_terms[min(index_group) + 1:max(index_group)]
            if term not in query_terms
        ])
        for index_group in index_groups
    ]

    occurrences_counter = Counter(non_query_term_occurrences)
    min_occurrences = min(occurrences_counter.keys())
    min_occurrences_count = occurrences_counter[min_occurrences]
    return min_occurrences, min_occurrences_count


def average_smallest_span(
        query_terms: Set[str],
        document_terms: List[str]
):
    return mean(
        max(group) - min(group)
        for group in query_term_index_groups(query_terms, document_terms)
    )
