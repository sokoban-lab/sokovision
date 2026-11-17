//! Union-find over appearance indices.

/// Disjoint sets with path halving, used to group appearances that look alike.
pub(super) struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    pub fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    pub fn find(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }

    pub fn union(&mut self, left: usize, right: usize) {
        let (left, right) = (self.find(left), self.find(right));
        if left != right {
            self.parent[right] = left;
        }
    }

    /// Every non-empty set, in root order. Each set keeps the order in which its
    /// members were given.
    pub fn groups(&mut self, members: impl IntoIterator<Item = usize>) -> Vec<Vec<usize>> {
        let mut groups = vec![Vec::new(); self.parent.len()];
        for index in members {
            let root = self.find(index);
            groups[root].push(index);
        }
        groups.retain(|group| !group.is_empty());
        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_transitively_and_keeps_order() {
        let mut set = DisjointSet::new(5);
        set.union(0, 3);
        set.union(3, 4);

        assert_eq!(set.find(4), set.find(0));
        assert_eq!(set.groups(0..5), vec![vec![0, 3, 4], vec![1], vec![2]]);
    }
}
