/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.classfile.visitor;

import proguard.classfile.*;

import java.util.*;
import java.util.concurrent.*;


/**
 * This ClassPoolVisitor will visit all Clazz objects of the class pool
 * in a parallel way. For each thread, a separate ClassVisitor will be
 * created using {@link ClassVisitorFactory#createClassVisitor()}.
 * <p>
 * The number of parallel threads is coupled to the number of available
 * processors:
 * <pre>
 *     parallel_threads = Runtime.getRuntime().availableProcessors() - 1;
 * </pre>
 * <p>
 * It is possible to override the number of threads by setting the
 * environment variable {@code parallel.threads} to an integer > 0.
 *
 * @author Thomas Neidhart
 */
public class ParallelAllClassVisitor
implements ClassPoolVisitor
{
    private static final int THREAD_COUNT;
    static {
        Integer threads = null;
        try {
            String threadCountString = System.getProperty("parallel.threads");
            if (threadCountString != null)
            {
                threads = Integer.parseInt(threadCountString);
            }
        }
        catch (Exception ex) {}

        threads = threads == null ?
            Runtime.getRuntime().availableProcessors() - 1 :
            Math.min(threads.intValue(), Runtime.getRuntime().availableProcessors());

        THREAD_COUNT = threads.intValue();
    }


    /**
     * A factory for ClassVisitor objects.
     */
    public interface ClassVisitorFactory
    {
        /**
         * Creates a ClassVisitor that will be used during
         * parallel visiting of classes in a ClassPool.
         */
        ClassVisitor createClassVisitor();
    }


    private final ClassVisitorFactory  classVisitorFactory;


    /**
     * Create a new ParallelAllClassVisitor that will use the given factory
     * to visit all classes in a ClassPool in a parallel way.
     */
    public ParallelAllClassVisitor(ClassVisitorFactory classVisitorFactory)
    {
        this.classVisitorFactory = classVisitorFactory;
    }


    // Implementations for ClassPoolVisitor.

    public void visitClassPool(ClassPool classPool)
    {
        if (THREAD_COUNT <= 1)
        {
            // Fallback to single thread execution if the thread count
            // was overridden by an environment variable.
            classPool.classesAccept(classVisitorFactory.createClassVisitor());
        }
        else
        {
            ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT, new MyThreadFactory());

            MyThreadedClassVisitor classVisitor = new MyThreadedClassVisitor(executor);

            classPool.classesAccept(classVisitor);

            try
            {
                // Shutdown the executor service to release memory.
                executor.shutdown();

                // Rethrow any exception that was thrown in the executor threads.
                classVisitor.awaitTermination();
            }
            catch (InterruptedException e)
            {
                throw new RuntimeException("Parallel execution is taking too long", e);
            }
            catch (ExecutionException e)
            {
                throw new RuntimeException(e.getCause());
            }
        }
    }


    private class MyThreadFactory
    implements ThreadFactory
    {
        private int threadCounter = 0;

        public Thread newThread(Runnable runnable)
        {
            return new MyClassVisitorThread(++threadCounter, runnable);
        }
    }


    private class MyClassVisitorThread
    extends       Thread
    {
        private final ClassVisitor classVisitor = classVisitorFactory.createClassVisitor();

        public MyClassVisitorThread(int counter, Runnable runnable)
        {
            super(runnable, "Parallel Class Visitor " + counter);
        }
    }


    private static class MyThreadedClassVisitor
    implements ClassVisitor
    {
        private final ExecutorService executorService;

        private final List<Future> futures = new ArrayList<Future>();

        public MyThreadedClassVisitor(ExecutorService executorService)
        {
            this.executorService = executorService;
        }

        public void awaitTermination() throws ExecutionException, InterruptedException
        {
            for (Future future : futures)
            {
                future.get();
            }
        }

        // Implementations for ClassVisitor.

        public void visitLibraryClass(LibraryClass libraryClass)
        {
            submitClassToExecutorService(libraryClass);
        }


        public void visitProgramClass(ProgramClass programClass)
        {
            submitClassToExecutorService(programClass);
        }


        private void submitClassToExecutorService(final Clazz clazz)
        {
            futures.add(executorService.submit(new Runnable()
            {
                public void run()
                {
                    MyClassVisitorThread thread = (MyClassVisitorThread)Thread.currentThread();
                    clazz.accept(thread.classVisitor);
                }
            }));
        }
    }
}
