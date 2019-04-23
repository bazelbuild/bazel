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
package proguard.ant;

import org.apache.tools.ant.types.DataType;
import proguard.classfile.util.ClassUtil;
import proguard.util.ListUtil;

import java.util.List;

/**
 * This DataType represents a name filter in Ant.
 *
 * @author Eric Lafortune
 */
public class FilterElement extends DataType
{
    private String filter;


    /**
     * Adds the contents of this element to the given name filter.
     * @param filter   the list of attributes to be extended.
     * @param internal specifies whether the filter string should be converted
     *                 to internal types.
     */
    public void appendTo(List filter, boolean internal)
    {
        // Get the referenced element, or else this one.
        FilterElement filterElement = isReference() ?
            (FilterElement)getCheckedRef(this.getClass(),
                                         this.getClass().getName()) :
            this;

        String filterString = filterElement.filter;

        if (filterString == null)
        {
            // Clear the filter to keep all names.
            filter.clear();
        }
        else
        {
            if (internal)
            {
                filterString = ClassUtil.internalClassName(filterString);
            }

            // Append the filter.
            filter.addAll(ListUtil.commaSeparatedList(filterString));
        }
    }


    // Ant task attributes.

    public void setName(String name)
    {
        this.filter = name;
    }


    public void setFilter(String filter)
    {
        this.filter = filter;
    }
}
