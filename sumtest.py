def twoSum(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        compliment = {}
        for x in nums:
            print(x)
            compliment[x] = target - x
        print(compliment)
        """
        {
            2 : 9-2 = 7,
            7 : 9-7 = 2,
            11 : 9 - 11 = -2
            15 : 9 - 15 = -6
        }
        for n, x in enumerate(compliment.keys()):
            if compliment[x] in nums:
                return nums[n]

        """
        i = 0
        for k, v in compliment.items():
            if v in nums:
                print(k, i, v)
            i += 1


def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """
    # ans = []

    # for n, x in enumerate(s):

    #     if ans[n] != s[x+1]:
    #         ans.append(s[n])
    ans = sorted(set(s), key = s.index)
    subsequenceCheck = ''.join(ans)
    if subsequenceCheck in s:
        return len(ans)

    subsequenceCheck = list(subsequenceCheck)
    while ''.join(subsequenceCheck) not in s:
        print(subsequenceCheck)
        subsequenceCheck = subsequenceCheck.pop()

import math
def findMedianSortedArrays(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    print(f"For {nums1} and {nums2}")
    merged = sorted(nums1 + nums2)
    if len(merged)%2 != 0:
        mp = math.floor(len(merged)/2)
        ans = merged[int(mp)]
        return ans
    if len(merged) % 2 == 0:
        print("hello")
        ml = int(len(merged)/2)
        mr = int(len(merged)/2 +1)
        return (ml+mr)/2
print(findMedianSortedArrays([1, 3] ,[2]))
print(findMedianSortedArrays([1, 2] ,[3, 4]))
