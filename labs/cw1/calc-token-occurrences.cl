__kernel void calcTokenOccurrences(
    __global const char* data,
    __global int* results,
    __global const char* tokens,
    __global const int* tokenOffsets,
    __global const int* tokenLengths,
    const int dataSize,
    const int numTokens) {
    int id = get_global_id(0);
    if (id >= dataSize) return; 
    for (int t = 0; t < numTokens; ++t) {
        int tokenLen = tokenLengths[t];
        int tokenIdx = tokenOffsets[t];
        int match = 1;
        for (int i = 0; i < tokenLen; ++i) {
            if (id + i >= dataSize || data[id + i] != tokens[tokenIdx + i]) {
                match = 0;
                break;
            }
        }
        if (!match) continue;
        int iPrefix = id - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z') continue;
        int iSuffix = id + tokenLen;
        if (iSuffix < dataSize && data[iSuffix] >= 'a' && data[iSuffix] <= 'z') continue;
        atomic_inc(&results[t]);
    }
}
 