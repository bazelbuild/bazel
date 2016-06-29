package org.checkerframework.dataflow.cfg;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.analysis.AbstractValue;
import org.checkerframework.dataflow.analysis.Analysis;
import org.checkerframework.dataflow.analysis.Store;
import org.checkerframework.dataflow.analysis.TransferFunction;
import org.checkerframework.javacutil.BasicTypeProcessor;
import org.checkerframework.javacutil.TreeUtils;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import javax.lang.model.element.ExecutableElement;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.xml.ws.Holder;

import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.util.TreePathScanner;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.List;

/**
 * Class to generate the DOT representation of the control flow graph of a given
 * method.
 *
 * @author Stefan Heule
 */
public class JavaSource2CFGDOT {

    /** Main method. */
    public static void main(String[] args) {
        if (args.length < 2) {
            printUsage();
            System.exit(1);
        }
        String input = args[0];
        String output = args[1];
        File file = new File(input);
        if (!file.canRead()) {
            printError("Cannot read input file: " + file.getAbsolutePath());
            printUsage();
            System.exit(1);
        }

        String method = "test";
        String clas = "Test";
        boolean pdf = false;
        boolean error = false;

        for (int i = 2; i < args.length; i++) {
            if (args[i].equals("-pdf")) {
                pdf = true;
            } else if (args[i].equals("-method")) {
                if (i >= args.length - 1) {
                    printError("Did not find <name> after -method.");
                    continue;
                }
                i++;
                method = args[i];
            } else if (args[i].equals("-class")) {
                if (i >= args.length - 1) {
                    printError("Did not find <name> after -class.");
                    continue;
                }
                i++;
                clas = args[i];
            } else {
                printError("Unknown command line argument: " + args[i]);
                error = true;
            }
        }

        if (error) {
            System.exit(1);
        }

        generateDOTofCFG(input, output, method, clas, pdf);
    }

    /** Print an error message. */
    protected static void printError(String string) {
        System.err.println("ERROR: " + string);
    }

    /** Print usage information. */
    protected static void printUsage() {
        System.out
                .println("Generate the control flow graph of a Java method, represented as a DOT graph.");
        System.out
                .println("Parameters: <inputfile> <outputdir> [-method <name>] [-class <name>] [-pdf]");
        System.out
                .println("    -pdf:    Also generate the PDF by invoking 'dot'.");
        System.out
                .println("    -method: The method to generate the CFG for (defaults to 'test').");
        System.out
                .println("    -class:  The class in which to find the method (defaults to 'Test').");
    }

    /** Just like method above but without analysis. */
    public static void generateDOTofCFG(String inputFile, String outputDir,
            String method, String clas, boolean pdf) {
        generateDOTofCFG(inputFile, outputDir, method, clas, pdf, null);
    }

    /**
     * Generate the DOT representation of the CFG for a method.
     *
     * @param inputFile
     *            Java source input file.
     * @param outputDir
     *            Source output directory.
     * @param method
     *            Method name to generate the CFG for.
     * @param pdf
     *            Also generate a PDF?
     * @param analysis
     *            Analysis to perform befor the visualization (or
     *            {@code null} if no analysis is to be performed).
     */
    public static
    <A extends AbstractValue<A>, S extends Store<S>, T extends TransferFunction<A, S>>
    void generateDOTofCFG(
            String inputFile, String outputDir, String method, String clas,
            boolean pdf, /*@Nullable*/ Analysis<A, S, T> analysis) {
        Entry<MethodTree, CompilationUnitTree> m = getMethodTreeAndCompilationUnit(inputFile, method, clas);
        generateDOTofCFG(inputFile, outputDir, method, clas, pdf, analysis, m.getKey(), m.getValue());
    }

    public static
    <A extends AbstractValue<A>, S extends Store<S>, T extends TransferFunction<A, S>>
    void generateDOTofCFG(
            String inputFile, String outputDir, String method, String clas,
            boolean pdf, /*@Nullable*/ Analysis<A, S, T> analysis, MethodTree m,
            CompilationUnitTree r) {
        String fileName = (new File(inputFile)).getName();
        System.out.println("Working on " + fileName + "...");

        if (m == null) {
            printError("Method not found.");
            System.exit(1);
        }

        ControlFlowGraph cfg = CFGBuilder.build(r, null, m, null);
        if (analysis != null) {
            analysis.performAnalysis(cfg);
        }

        Map<String, Object> args = new HashMap<>();
        args.put("outdir", outputDir);
        args.put("checkerName", "");

        CFGVisualizer<A, S, T> viz = new DOTCFGVisualizer<A, S, T>();
        viz.init(args);
        Map<String, Object> res = viz.visualize(cfg, cfg.getEntryBlock(), analysis);
        viz.shutdown();

        if (pdf) {
            producePDF((String) res.get("dotFileName"));
        }
    }

    /**
     * Invoke DOT to generate a PDF.
     */
    protected static void producePDF(String file) {
        try {
            String command = "dot -Tpdf \"" + file + ".txt\" -o \"" + file
                    + ".pdf\"";
            Process child = Runtime.getRuntime().exec(command);
            child.waitFor();
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * @return the AST of a specific method in a specific class in a specific
     *         file (or null if no such method exists)
     */
    public static /*@Nullable*/ MethodTree getMethodTree(String file,
            final String method, String clas) {
        return getMethodTreeAndCompilationUnit(file, method, clas).getKey();
    }

    /**
     * @return the AST of a specific method in a specific class as well as the
     *         {@link CompilationUnitTree} in a specific file (or null they do
     *         not exist).
     */
    public static Entry</*@Nullable*/ MethodTree, /*@Nullable*/ CompilationUnitTree> getMethodTreeAndCompilationUnit(
            String file, final String method, String clas) {
        final Holder<MethodTree> m = new Holder<>();
        final Holder<CompilationUnitTree> c = new Holder<>();
        BasicTypeProcessor typeProcessor = new BasicTypeProcessor() {
            @Override
            protected TreePathScanner<?, ?> createTreePathScanner(
                    CompilationUnitTree root) {
                c.value = root;
                return new TreePathScanner<Void, Void>() {
                    @Override
                    public Void visitMethod(MethodTree node, Void p) {
                        ExecutableElement el = TreeUtils
                                .elementFromDeclaration(node);
                        if (el.getSimpleName().contentEquals(method)) {
                            m.value = node;
                            // stop execution by throwing an exception. this
                            // makes sure that compilation does not proceed, and
                            // thus the AST is not modified by further phases of
                            // the compilation (and we save the work to do the
                            // compilation).
                            throw new RuntimeException();
                        }
                        return null;
                    }
                };
            }
        };

        Context context = new Context();
        JavaCompiler javac = new JavaCompiler(context);
        javac.attrParseOnly = true;
        JavacFileManager fileManager = (JavacFileManager) context
                .get(JavaFileManager.class);

        JavaFileObject l = fileManager
                .getJavaFileObjectsFromStrings(List.of(file)).iterator().next();

        PrintStream err = System.err;
        try {
            // redirect syserr to nothing (and prevent the compiler from issuing
            // warnings about our exception.
            System.setErr(new PrintStream(new OutputStream() {
                @Override
                public void write(int b) throws IOException {
                }
            }));
            javac.compile(List.of(l), List.of(clas), List.of(typeProcessor));
        } catch (Throwable e) {
            // ok
        } finally {
            System.setErr(err);
        }
        return new Entry<MethodTree, CompilationUnitTree>() {
            @Override
            public CompilationUnitTree setValue(CompilationUnitTree value) {
                return null;
            }

            @Override
            public CompilationUnitTree getValue() {
                return c.value;
            }

            @Override
            public MethodTree getKey() {
                return m.value;
            }
        };
    }

}
