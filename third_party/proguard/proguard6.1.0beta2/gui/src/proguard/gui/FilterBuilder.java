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
package proguard.gui;

import javax.swing.*;

/**
 * This class builds filters corresponding to the selections and names of a
 * given list of check boxes.
 */
public class FilterBuilder
{
    private JCheckBox[] checkBoxes;
    private char        separator;


    /**
     * Creates a new FilterBuilder.
     * @param checkBoxes the check boxes with names and selections that should
     *                   be reflected in the output filter.
     * @param separator  the separator for the names in the check boxes.
     */
    public FilterBuilder(JCheckBox[] checkBoxes, char separator)
    {
        this.checkBoxes = checkBoxes;
        this.separator  = separator;
    }


    /**
     * Builds a filter for the current names and selections of the check boxes.
     */
    public String buildFilter()
    {
        StringBuffer positive = new StringBuffer();
        StringBuffer negative = new StringBuffer();

        buildFilter("", positive, negative);

        return positive.length() <= negative.length() ?
            positive.toString() :
            negative.toString();
    }


    /**
     * Builds two versions of the filter for the given prefix.
     * @param prefix   the prefix.
     * @param positive the filter to be extended, assuming the matching
     *                 strings are accepted.
     * @param negative the filter to be extended, assuming the matching
     *                 strings are rejected.
     */
    private void buildFilter(String       prefix,
                             StringBuffer positive,
                             StringBuffer negative)
    {
        int positiveCount = 0;
        int negativeCount = 0;

        // Count all selected and unselected check boxes with the prefix.
        for (int index = 0; index < checkBoxes.length; index++)
        {
            JCheckBox checkBox = checkBoxes[index];
            String    name     = checkBox.getText();

            if (name.startsWith(prefix))
            {
                if (checkBox.isSelected())
                {
                    positiveCount++;
                }
                else
                {
                    negativeCount++;
                }
            }
        }

        // Are there only unselected check boxes?
        if (positiveCount == 0)
        {
            // Extend the positive filter with exceptions and return.
            if (positive.length() > 0)
            {
                positive.append(',');
            }
            positive.append('!').append(prefix);
            if (prefix.length() == 0 ||
                prefix.charAt(prefix.length()-1) == separator)
            {
                positive.append('*');
            }

            return;
        }

        // Are there only selected check boxes?
        if (negativeCount == 0)
        {
            // Extend the negative filter with exceptions and return.
            if (negative.length() > 0)
            {
                negative.append(',');
            }
            negative.append(prefix);
            if (prefix.length() == 0 ||
                prefix.charAt(prefix.length()-1) == separator)
            {
                negative.append('*');
            }

            return;
        }

        // Create new positive and negative filters for names starting with the
        // prefix only.
        StringBuffer positiveFilter = new StringBuffer();
        StringBuffer negativeFilter = new StringBuffer();

        String newPrefix = null;

        for (int index = 0; index < checkBoxes.length; index++)
        {
            String name = checkBoxes[index].getText();

            if (name.startsWith(prefix))
            {
                if (newPrefix == null ||
                    !name.startsWith(newPrefix))
                {
                    int prefixIndex =
                        name.indexOf(separator, prefix.length()+1);

                    newPrefix = prefixIndex >= 0 ?
                        name.substring(0, prefixIndex+1) :
                        name;

                    buildFilter(newPrefix,
                                positiveFilter,
                                negativeFilter);
                }
            }
        }

        // Extend the positive filter.
        if (positiveFilter.length() <= negativeFilter.length() + prefix.length() + 3)
        {
            if (positive.length() > 0 &&
                positiveFilter.length() > 0)
            {
                positive.append(',');
            }

            positive.append(positiveFilter);
        }
        else
        {
            if (positive.length() > 0 &&
                negativeFilter.length() > 0)
            {
                positive.append(',');
            }

            positive.append(negativeFilter).append(",!").append(prefix).append('*');
        }

        // Extend the negative filter.
        if (negativeFilter.length() <= positiveFilter.length() + prefix.length() + 4)
        {
            if (negative.length() > 0 &&
                negativeFilter.length() > 0)
            {
                negative.append(',');
            }

            negative.append(negativeFilter);
        }
        else
        {
            if (negative.length() > 0 &&
                positiveFilter.length() > 0)
            {
                negative.append(',');
            }

            negative.append(positiveFilter).append(',').append(prefix).append('*');
        }
    }
}
