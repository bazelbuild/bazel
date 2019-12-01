package com.github.luben.zstd;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class ZstdDictTrainer {
    private final int allocatedSize;
    private final ByteBuffer trainingSamples;
    private final List<Integer> sampleSizes;
    private final int dictSize;
    private long filledSize;

    public ZstdDictTrainer(int sampleSize, int dictSize) {
        trainingSamples = ByteBuffer.allocateDirect(sampleSize);
        sampleSizes =  new ArrayList<Integer>();
        this.allocatedSize = sampleSize;
        this.dictSize = dictSize;
    }

    public synchronized boolean addSample(byte[] sample) {
        if (filledSize + sample.length > allocatedSize) {
            return false;
        }
        trainingSamples.put(sample);
        sampleSizes.add(sample.length);
        filledSize += sample.length;
        return true;
    }

    public ByteBuffer trainSamplesDirect() throws ZstdException {
        return trainSamplesDirect(false);
    }

    public synchronized ByteBuffer trainSamplesDirect(boolean legacy) throws ZstdException {
        ByteBuffer dictBuffer = ByteBuffer.allocateDirect(dictSize);
        long l = Zstd.trainFromBufferDirect(trainingSamples, copyToIntArray(sampleSizes), dictBuffer, legacy);
        if (Zstd.isError(l)) {
            dictBuffer.limit(0);
            throw new ZstdException(l);
        }
        dictBuffer.limit(Long.valueOf(l).intValue());
        return dictBuffer;
    }

    public byte[] trainSamples() throws ZstdException {
        return trainSamples(false);
    }

    public byte[] trainSamples(boolean legacy) throws ZstdException {
        ByteBuffer byteBuffer = trainSamplesDirect(legacy);
        byte[] bytes = new byte[byteBuffer.remaining()];
        byteBuffer.get(bytes);
        return bytes;
    }

    private int[] copyToIntArray(List<Integer> list) {
        int[] ints = new int[list.size()];
        int idx = 0;
        for (Integer i: list) {
            ints[idx] = i;
            idx++;
        }
        return ints;
    }
}
