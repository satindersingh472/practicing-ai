function two_sum(nums,target){
        num1 = nums[0]
        for (let i=0;i<nums.length;i++){
            if(nums[i] + num1 === target){
                result_one =  nums.indexOf(num1)
                result_two = nums.indexOf(nums[i])
                return [result_one,result_two]
            } else {
                num1 = nums[i]                             
            }
        }
}
 answer = two_sum([1,2,3],4)
 console.log(answer)