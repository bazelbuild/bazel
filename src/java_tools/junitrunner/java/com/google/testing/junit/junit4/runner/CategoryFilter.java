package com.google.testing.junit.junit4.runner;

import org.junit.experimental.categories.Category;
import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

import java.util.Set;
import java.util.StringJoiner;

/**
 * A filter that can be used to select tests and suites based on {@link Category} annotations. It will match any tests
 * that have at least one included category and no excluded categories. If the user has not specified categories to
 * include, it will allow any tests that are not explicitly excluded.
 */
public class CategoryFilter extends Filter {

    private final Set<Class<?>> included, excluded;

    public CategoryFilter(Set<Class<?>> included, Set<Class<?>> excluded) {
        this.included = included;
        this.excluded = excluded;
    }

    @Override
    public boolean shouldRun(Description description) {
        Category annotation = description.getAnnotation(Category.class);
        // Is the object explicitly excluded?
        if (isExcluded(annotation)) {
            return false;
        }
        // Is the object a test whose parent class is explicitly included or excluded?
        if (description.isTest()) {
            Class<?> testClass = description.getTestClass();
            if (testClass != null) {
                Category parentAnnotation = testClass.getAnnotation(Category.class);
                if (isExcluded(parentAnnotation)) {
                    return false;
                }
                if (isIncluded(parentAnnotation)) {
                    return true;
                }
            }
        } else {
            // Is the object a test suite that contains any explicitly included tests?
            for (Description child : description.getChildren()) {
                Category childAnnotation = child.getAnnotation(Category.class);
                if (isIncluded(childAnnotation) && !isExcluded(childAnnotation)) {
                    return true;
                }
            }
        }
        // Check this last, as tests may be excluded by their parent classes.
        return isIncluded(annotation);
    }

    private boolean isIncluded(Category annotation) {
        if (included.isEmpty()) {
            // If the user has not specified any included categories, we should default to a permissive mode
            // where any tests that have not explicitly been excluded are run. Otherwise, no tests would run.
            return true;
        }
        return contains(included, annotation);
    }

    private boolean isExcluded(Category annotation) {
        if (excluded.isEmpty()) {
            return false;
        }
        return contains(excluded, annotation);
    }

    private boolean contains(Set<Class<?>> set, Category annotation) {
        if (annotation == null) {
            return false;
        }
        for (Class<?> clazz : annotation.value()) {
            if (set.contains(clazz)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String describe() {
        return toString();
    }

    private String listClasses(String prefix, Iterable<Class<?>> classes) {
        StringJoiner j = new StringJoiner(", ", prefix + " [", "]");
        for (Class<?> c : classes) {
            j.add(c.getSimpleName());
        }
        return j.toString();
    }

    @Override
    public String toString() {
        return (included.isEmpty() ? "all tests" : listClasses("categories", included))
             + (excluded.isEmpty() ? "" : listClasses(" excluding categories", excluded));
    }
}
