package org.checkerframework.dataflow.cfg.playground;

import org.checkerframework.dataflow.analysis.Analysis;
import org.checkerframework.dataflow.cfg.JavaSource2CFGDOT;
import org.checkerframework.dataflow.constantpropagation.Constant;
import org.checkerframework.dataflow.constantpropagation.ConstantPropagationStore;
import org.checkerframework.dataflow.constantpropagation.ConstantPropagationTransfer;

public class ConstantPropagationPlayground {

    /**
     * Run constant propagation for a specific file and create a PDF of the CFG
     * in the end.
     */
    public static void main(String[] args) {

        /* Configuration: change as appropriate */
        String inputFile = "cfg-input.java"; // input file name and path
        String outputDir = "cfg"; // output directory
        String method = "test"; // name of the method to analyze
        String clazz = "Test"; // name of the class to consider

        // run the analysis and create a PDF file
        ConstantPropagationTransfer transfer = new ConstantPropagationTransfer();
        // TODO: correct processing environment
        Analysis<Constant, ConstantPropagationStore, ConstantPropagationTransfer> analysis = new Analysis<>(
                null, transfer);
        JavaSource2CFGDOT.generateDOTofCFG(inputFile, outputDir, method,
                clazz, true, analysis);
    }

}
