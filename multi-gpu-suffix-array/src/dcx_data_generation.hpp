#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// generates random text over alphabet {'b', 'c', 'd', 'e'}
// 'a' is used as a sentinel character at the end
std::string generate_random_text(std::size_t n, unsigned seed = 0) {
  std::mt19937 rng(seed ? seed : std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 3);
  const char alphabet[4] = { 'b', 'c', 'd', 'e' };
  std::string s;
  s.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
    s.push_back(alphabet[dist(rng)]);
  for (std::size_t i = 0; i < 42; ++i) {
    s.push_back('a');
  }
  return s;
}

// get length-7 prefix for suffix starting at pos
auto prefix7_of(const std::string& s, std::size_t pos) {
  std::array<unsigned char, DCX::X> pref;
  for (std::size_t i = 0; i < DCX::X; ++i) {
    pref[i] = s[pos + i];
  }
  return pref;
}

// return indices of suffixes in lex. order
std::vector<std::size_t> naive_suffix_sort(std::size_t n,
  std::string const& text) {
  std::vector<std::size_t> idx(n);
  for (std::size_t i = 0; i < n; ++i)
    idx[i] = i;
  // comparator comparing suffixes text[i..] < text[j..]
  auto suffix_cmp = [&text](std::size_t a, std::size_t b) {
    std::size_t na = text.size() - a;
    std::size_t nb = text.size() - b;
    std::size_t k = 0;
    // compare character by character
    while (k < na && k < nb) {
      char ca = text[a + k];
      char cb = text[b + k];
      if (ca < cb)
        return true;
      if (ca > cb)
        return false;
      ++k;
    }
    // if one is prefix of the other, shorter is smaller
    return na < nb;
    };
  std::sort(idx.begin(), idx.end(), suffix_cmp);
  return idx;
}
std::vector<std::size_t> naive_suffix_sort(std::size_t n,
  unsigned char* text) {
  std::vector<std::size_t> idx(n);
  for (std::size_t i = 0; i < n; ++i)
    idx[i] = i;
  // comparator comparing suffixes text[i..] < text[j..]
  auto suffix_cmp = [text, n](std::size_t a, std::size_t b) {
    std::size_t na = n - a;
    std::size_t nb = n - b;
    std::size_t k = 0;
    // compare character by character
    while (k < na && k < nb) {
      char ca = text[a + k];
      char cb = text[b + k];
      if (ca < cb)
        return true;
      if (ca > cb)
        return false;
      ++k;
    }
    // if one is prefix of the other, shorter is smaller
    return na < nb;
    };
  std::sort(idx.begin(), idx.end(), suffix_cmp);
  return idx;
}
auto generate_data_dcx(std::size_t n, std::size_t seed) {
  std::string text = generate_random_text(n, seed);
  auto idx = naive_suffix_sort(n, text);

  // compute rank: rank[pos] = rank of suffix starting at pos (0-based: 0 =
  // smallest suffix)
  std::vector<std::size_t> rank(n);
  for (std::size_t r = 0; r < n; ++r) {
    rank[idx[r]] = r;
  }

  // Build tuples: (prefix7, index, rank) for each suffix (index 0..n-1)
  std::vector<MergeSuffixes> tuples;
  tuples.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    MergeSuffixes t;
    t.prefix = prefix7_of(text, i);
    t.index = i;
    // compute 3 ranks of DC sample positions >= current suffix
    std::size_t counter = 0;
    for (std::size_t o = 0; o < 8 && counter < 3; ++o) {
      const bool mod1 = (i + o) % 7 == 1;
      const bool mod2 = (i + o) % 7 == 2;
      const bool mod4 = (i + o) % 7 == 4;
      if (mod1 || mod2 || mod4) {
        t.ranks[counter] = (i + o >= n) ? 0 : (rank[i + o] + 1);
        counter++;
      }
    }
    tuples.push_back(t);
  }
  return std::make_tuple(text, tuples);
}
