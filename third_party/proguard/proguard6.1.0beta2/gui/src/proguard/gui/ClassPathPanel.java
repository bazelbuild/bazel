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

import proguard.*;
import proguard.util.ListUtil;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.List;

/**
 * This <code>ListPanel</code> allows the user to add, edit, filter, move, and
 * remove ClassPathEntry objects in a ClassPath object.
 *
 * @author Eric Lafortune
 */
class ClassPathPanel extends ListPanel
{
    private final JFrame       owner;
    private final boolean      inputAndOutput;
    private final JFileChooser chooser;
    private final FilterDialog filterDialog;


    public ClassPathPanel(JFrame owner, boolean inputAndOutput)
    {
        super();

        super.firstSelectionButton = inputAndOutput ? 3 : 2;

        this.owner          = owner;
        this.inputAndOutput = inputAndOutput;

        list.setCellRenderer(new MyListCellRenderer());

        chooser = new JFileChooser("");
        chooser.setMultiSelectionEnabled(true);
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        chooser.addChoosableFileFilter(
            new ExtensionFileFilter(msg("jarExtensions"),
                                    new String[] { ".apk", ".ap_", ".jar", ".aar", ".war", ".ear", ".jmod", ".zip" }));
        chooser.setApproveButtonText(msg("ok"));

        filterDialog = new FilterDialog(owner, msg("enterFilter"));

        addAddButton(inputAndOutput, false);
        if (inputAndOutput)
        {
            addAddButton(inputAndOutput, true);
        }
        addEditButton();
        addFilterButton();
        addRemoveButton();
        addUpButton();
        addDownButton();

        enableSelectionButtons();
    }


    protected void addAddButton(boolean       inputAndOutput,
                                final boolean isOutput)
    {
        JButton addButton = new JButton(msg(inputAndOutput ?
                                            isOutput       ? "addOutput" :
                                                             "addInput" :
                                                             "add"));
        addButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                chooser.setDialogTitle(msg("addJars"));
                chooser.setSelectedFile(null);
                chooser.setSelectedFiles(null);

                int returnValue = chooser.showOpenDialog(owner);
                if (returnValue == JFileChooser.APPROVE_OPTION)
                {
                    File[] selectedFiles = chooser.getSelectedFiles();
                    ClassPathEntry[] entries = classPathEntries(selectedFiles, isOutput);

                    // Add the new elements.
                    addElements(entries);
                }
            }
        });

        addButton(tip(addButton, inputAndOutput ?
                                 isOutput       ? "addOutputTip" :
                                                  "addInputTip" :
                                                  "addTip"));
    }


    protected void addEditButton()
    {
        JButton editButton = new JButton(msg("edit"));
        editButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                boolean isOutput = false;

                int[] selectedIndices = list.getSelectedIndices();

                // Copy the Object array into a File array.
                File[] selectedFiles = new File[selectedIndices.length];
                for (int index = 0; index < selectedFiles.length; index++)
                {
                    ClassPathEntry entry =
                        (ClassPathEntry)listModel.getElementAt(selectedIndices[index]);

                    isOutput = entry.isOutput();

                    selectedFiles[index] = entry.getFile();
                }

                chooser.setDialogTitle(msg("chooseJars"));

                // Up to JDK 1.3.1, setSelectedFiles doesn't show in the file
                // chooser, so we just use setSelectedFile first. It also sets
                // the current directory.
                chooser.setSelectedFile(selectedFiles[0].getAbsoluteFile());
                chooser.setSelectedFiles(selectedFiles);

                int returnValue = chooser.showOpenDialog(owner);
                if (returnValue == JFileChooser.APPROVE_OPTION)
                {
                    selectedFiles = chooser.getSelectedFiles();
                    ClassPathEntry[] entries = classPathEntries(selectedFiles, isOutput);

                    // If there are the same number of files selected now as
                    // there were before, we can just replace the old ones.
                    if (selectedIndices.length == selectedFiles.length)
                    {
                        // Replace the old elements.
                        setElementsAt(entries, selectedIndices);
                    }
                    else
                    {
                        // Remove the old elements.
                        removeElementsAt(selectedIndices);

                        // Add the new elements.
                        addElements(entries);
                    }
                }
            }
        });

        addButton(tip(editButton, "editTip"));
    }


    protected void addFilterButton()
    {
        JButton filterButton = new JButton(msg("filter"));
        filterButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                if (!list.isSelectionEmpty())
                {
                    int[] selectedIndices = list.getSelectedIndices();

                    // Put the filters of the first selected entry in the dialog.
                    getFiltersFrom(selectedIndices[0]);

                    int returnValue = filterDialog.showDialog();
                    if (returnValue == FilterDialog.APPROVE_OPTION)
                    {
                        // Apply the entered filters to all selected entries.
                        setFiltersAt(selectedIndices);
                    }
                }
            }
        });

        addButton(tip(filterButton, "filterTip"));
    }


    /**
     * Sets the ClassPath to be represented in this panel.
     */
    public void setClassPath(ClassPath classPath)
    {
        listModel.clear();

        if (classPath != null)
        {
            for (int index = 0; index < classPath.size(); index++)
            {
                listModel.addElement(classPath.get(index));
            }
        }

        // Make sure the selection buttons are properly enabled,
        // since the clear method doesn't seem to notify the listener.
        enableSelectionButtons();
    }


    /**
     * Returns the ClassPath currently represented in this panel.
     */
    public ClassPath getClassPath()
    {
        int size = listModel.size();
        if (size == 0)
        {
            return null;
        }

        ClassPath classPath = new ClassPath();
        for (int index = 0; index < size; index++)
        {
            classPath.add((ClassPathEntry)listModel.get(index));
        }

        return classPath;
    }


    /**
     * Converts the given array of File objects into a corresponding array of
     * ClassPathEntry objects.
     */
    private ClassPathEntry[] classPathEntries(File[] files, boolean isOutput)
    {
        ClassPathEntry[] entries = new ClassPathEntry[files.length];
        for (int index = 0; index < entries.length; index++)
        {
            entries[index] = new ClassPathEntry(files[index], isOutput);
        }
        return entries;
    }


    /**
     * Sets up the filter dialog with the filters from the specified class path
     * entry.
     */
    private void getFiltersFrom(int index)
    {
        ClassPathEntry firstEntry = (ClassPathEntry)listModel.get(index);

        filterDialog.setFilter(firstEntry.getFilter());
        filterDialog.setApkFilter(firstEntry.getApkFilter());
        filterDialog.setJarFilter(firstEntry.getJarFilter());
        filterDialog.setAarFilter(firstEntry.getAarFilter());
        filterDialog.setWarFilter(firstEntry.getWarFilter());
        filterDialog.setEarFilter(firstEntry.getEarFilter());
        filterDialog.setJmodFilter(firstEntry.getJmodFilter());
        filterDialog.setZipFilter(firstEntry.getZipFilter());
    }


    /**
     * Applies the entered filter to the specified class path entries.
     * Any previously set filters are discarded.
     */
    private void setFiltersAt(int[] indices)
    {
        for (int index = indices.length - 1; index >= 0; index--)
        {
            ClassPathEntry entry = (ClassPathEntry)listModel.get(indices[index]);
            entry.setFilter(filterDialog.getFilter());
            entry.setApkFilter(filterDialog.getApkFilter());
            entry.setJarFilter(filterDialog.getJarFilter());
            entry.setAarFilter(filterDialog.getAarFilter());
            entry.setWarFilter(filterDialog.getWarFilter());
            entry.setEarFilter(filterDialog.getEarFilter());
            entry.setJmodFilter(filterDialog.getJmodFilter());
            entry.setZipFilter(filterDialog.getZipFilter());
        }

        // Make sure they are selected and thus repainted.
        list.setSelectedIndices(indices);
    }


    /**
     * Attaches the tool tip from the GUI resources that corresponds to the
     * given key, to the given component.
     */
    private static JComponent tip(JComponent component, String messageKey)
    {
        component.setToolTipText(msg(messageKey));

        return component;
    }


    /**
     * Returns the message from the GUI resources that corresponds to the given
     * key.
     */
    private static String msg(String messageKey)
    {
         return GUIResources.getMessage(messageKey);
    }


    /**
     * This ListCellRenderer renders ClassPathEntry objects.
     */
    private class MyListCellRenderer implements ListCellRenderer
    {
        private static final String ARROW_IMAGE_FILE = "arrow.gif";

        private final JPanel cellPanel    = new JPanel(new GridBagLayout());
        private final JLabel iconLabel    = new JLabel("", JLabel.RIGHT);
        private final JLabel jarNameLabel = new JLabel("", JLabel.RIGHT);
        private final JLabel filterLabel  = new JLabel("", JLabel.RIGHT);

        private final Icon arrowIcon;


        public MyListCellRenderer()
        {
            GridBagConstraints jarNameLabelConstraints = new GridBagConstraints();
            jarNameLabelConstraints.anchor             = GridBagConstraints.WEST;
            jarNameLabelConstraints.insets             = new Insets(1, 2, 1, 2);

            GridBagConstraints filterLabelConstraints  = new GridBagConstraints();
            filterLabelConstraints.gridwidth           = GridBagConstraints.REMAINDER;
            filterLabelConstraints.fill                = GridBagConstraints.HORIZONTAL;
            filterLabelConstraints.weightx             = 1.0;
            filterLabelConstraints.anchor              = GridBagConstraints.EAST;
            filterLabelConstraints.insets              = jarNameLabelConstraints.insets;

            arrowIcon = new ImageIcon(Toolkit.getDefaultToolkit().getImage(this.getClass().getResource(ARROW_IMAGE_FILE)));

            cellPanel.add(iconLabel,    jarNameLabelConstraints);
            cellPanel.add(jarNameLabel, jarNameLabelConstraints);
            cellPanel.add(filterLabel,  filterLabelConstraints);
        }


        // Implementations for ListCellRenderer.

        public Component getListCellRendererComponent(JList   list,
                                                      Object  value,
                                                      int     index,
                                                      boolean isSelected,
                                                      boolean cellHasFocus)
        {
            ClassPathEntry entry = (ClassPathEntry)value;

            // Prepend an arrow to the output entries.
            if (inputAndOutput && entry.isOutput())
            {
                iconLabel.setIcon(arrowIcon);
            }
            else
            {
                iconLabel.setIcon(null);
            }

            // Set the entry name text.
            jarNameLabel.setText(entry.getName());

            // Set the filter text.
            StringBuffer filter = null;
            filter = appendFilter(filter, entry.getZipFilter());
            filter = appendFilter(filter, entry.getJmodFilter());
            filter = appendFilter(filter, entry.getEarFilter());
            filter = appendFilter(filter, entry.getWarFilter());
            filter = appendFilter(filter, entry.getAarFilter());
            filter = appendFilter(filter, entry.getJarFilter());
            filter = appendFilter(filter, entry.getApkFilter());
            filter = appendFilter(filter, entry.getFilter());

            if (filter != null)
            {
                filter.append(')');
            }

            filterLabel.setText(filter != null ? filter.toString() : "");

            // Set the colors.
            if (isSelected)
            {
                cellPanel.setBackground(list.getSelectionBackground());
                jarNameLabel.setForeground(list.getSelectionForeground());
                filterLabel.setForeground(list.getSelectionForeground());
            }
            else
            {
                cellPanel.setBackground(list.getBackground());
                jarNameLabel.setForeground(list.getForeground());
                filterLabel.setForeground(list.getForeground());
            }

            // Make the font color red if this is an input file that can't be read.
            if (!(inputAndOutput && entry.isOutput()) &&
                !entry.getFile().canRead())
            {
                jarNameLabel.setForeground(Color.red);
            }

            cellPanel.setOpaque(true);

            return cellPanel;
        }


        private StringBuffer appendFilter(StringBuffer filter, List additionalFilter)
        {
            if (filter != null)
            {
                filter.append(';');
            }

            if (additionalFilter != null)
            {
                if (filter == null)
                {
                    filter = new StringBuffer().append('(');
                }

                filter.append(ListUtil.commaSeparatedString(additionalFilter, true));
            }

            return filter;
        }
    }
}
